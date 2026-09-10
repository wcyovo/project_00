using System.Collections.Generic;
using UnityEngine;
using SmartBed.Data;

namespace SmartBed.Visualization
{
    /// <summary>
    /// 3D 床垫支撑形变视图（R1）：床垫网格高度由两个因素共同决定 ——
    /// ① 压力：人体压下去（越压越凹）；
    /// ② 气垫充气量：该气垫区抬升（充气多→抬得高，体现“各区气垫不同高度/软硬度”）。
    /// 表面贴压力热力图颜色，独立相机渲染到 RenderTexture 供 UI 显示。
    /// </summary>
    public class Bed3DView : MonoBehaviour
    {
        private const int BedLayer = 30;
        private static readonly int[] AirbagIds = { 12, 13, 40, 41, 42, 64, 65, 66 };

        public int Rows = BedConfig.Rows;
        public int Cols = BedConfig.Cols;
        public float cellSize = 0.45f;      // 每格世界尺寸
        public float maxIndent = 1.8f;       // 满压时的最大下凹深度
        public float maxLift = 0.9f;         // 气垫满充时的最大抬升高度
        public RectInt viewSize = new RectInt(0, 0, 512, 256);

        private Mesh mesh;
        private MeshFilter mf;
        private Material mat;
        private Camera cam;
        private RenderTexture rt;
        private Vector3[] verts;
        private Vector2[] uvs;
        private Texture2D heatCache;
        private int[] cellZoneId;            // 每格所属气垫区索引（-1 = 不属于任何气垫）
        private readonly float[] airbagLevels = new float[8];

        public RenderTexture RenderTexture => rt;

        private void Awake()
        {
            Build();
        }

        private void OnDestroy()
        {
            if (rt != null) rt.Release();
        }

        private void Build()
        {
            int n = Rows * Cols;
            verts = new Vector3[n];
            uvs = new Vector2[n];
            int[] tris = new int[(Rows - 1) * (Cols - 1) * 6];

            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Cols; c++)
                {
                    int i = r * Cols + c;
                    verts[i] = new Vector3(c * cellSize, 0f, r * cellSize);
                    uvs[i] = new Vector2((float)c / (Cols - 1), (float)r / (Rows - 1));
                }
            }

            // 预计算每格所属气垫区（用于按充气量抬升）
            cellZoneId = new int[n];
            for (int i = 0; i < n; i++)
            {
                cellZoneId[i] = -1;
                int rr = i / Cols, cc = i % Cols;
                for (int k = 0; k < AirbagIds.Length; k++)
                {
                    if (ZoneMap.IsInZone(AirbagIds[k], rr, cc)) { cellZoneId[i] = k; break; }
                }
            }

            int t = 0;
            for (int r = 0; r < Rows - 1; r++)
            {
                for (int c = 0; c < Cols - 1; c++)
                {
                    int i = r * Cols + c;
                    // 逆时针绕序，使表面法线朝上（相机从上往下才能看到）
                    tris[t++] = i;
                    tris[t++] = i + Cols;
                    tris[t++] = i + 1;
                    tris[t++] = i + 1;
                    tris[t++] = i + Cols;
                    tris[t++] = i + 1 + Cols;
                }
            }

            mesh = new Mesh();
            mesh.vertices = verts;
            mesh.uv = uvs;
            mesh.triangles = tris;
            mesh.RecalculateNormals();

            gameObject.layer = BedLayer;
            mf = gameObject.AddComponent<MeshFilter>();
            mf.sharedMesh = mesh;

            var mr = gameObject.AddComponent<MeshRenderer>();
            Shader sh = Shader.Find("Universal Render Pipeline/Lit");
            if (sh == null) sh = Shader.Find("Universal Render Pipeline/Unlit");
            mat = sh != null ? new Material(sh) : new Material(Shader.Find("Unlit/Color"));
            mr.sharedMaterial = mat;
            mat.SetColor("_BaseColor", Color.white);

            // 居中床垫
            transform.position = new Vector3(-(Cols - 1) * cellSize * 0.5f, 0f, -(Rows - 1) * cellSize * 0.5f);

            // 独立相机
            var camGo = new GameObject("BedCam");
            camGo.layer = BedLayer;
            cam = camGo.AddComponent<Camera>();
            cam.cullingMask = 1 << BedLayer;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.04f, 0.05f, 0.08f);
            cam.fieldOfView = 46f;
            rt = new RenderTexture(viewSize.width, viewSize.height, 16);
            cam.targetTexture = rt;
            cam.transform.position = new Vector3(9f, 15f, 16f);
            cam.transform.LookAt(new Vector3(0f, -1f, 0f));
        }

        /// <summary>更新床垫状态：压力(下凹) + 气垫充气(抬升)。</summary>
        public void UpdateState(float[] matrix, List<Airbag> airbags)
        {
            // 1) 更新各气垫充气量
            if (airbags != null)
            {
                for (int k = 0; k < AirbagIds.Length; k++)
                {
                    airbagLevels[k] = 0f;
                    for (int j = 0; j < airbags.Count; j++)
                    {
                        if (airbags[j].id == AirbagIds[k]) { airbagLevels[k] = airbags[j].level; break; }
                    }
                }
            }

            if (matrix == null || mesh == null) return;
            if (matrix.Length < Rows * Cols) return;

            // 2) 高度 = -压力下凹 + 气垫抬升
            for (int i = 0; i < verts.Length; i++)
            {
                float p = Mathf.Clamp01(matrix[i] / BedConfig.MaxPressure);
                float lift = 0f;
                int z = (cellZoneId != null && i < cellZoneId.Length) ? cellZoneId[i] : -1;
                if (z >= 0) lift = (airbagLevels[z] / 100f) * maxLift;
                verts[i].y = -p * maxIndent + lift;
            }
            mesh.vertices = verts;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();

            heatCache = PressureHeatmap.Build(heatCache, matrix);
            if (mat != null && heatCache != null) mat.SetTexture("_BaseMap", heatCache);
        }
    }
}
