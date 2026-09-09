using UnityEngine;
using SmartBed.Data;

namespace SmartBed.Visualization
{
    /// <summary>
    /// 3D 床垫支撑形变视图（R1 进阶）：根据 44x24 压力矩阵使床垫网格下凹，
    /// 下凹深度正比于该处压力，表面贴上压力热力图颜色，直观展示“支撑效果”。
    /// 独立相机只渲染本网格到 RenderTexture，供 UI RawImage 显示。
    /// </summary>
    public class Bed3DView : MonoBehaviour
    {
        private const int BedLayer = 30;

        public int Rows = BedConfig.Rows;
        public int Cols = BedConfig.Cols;
        public float cellSize = 0.35f;      // 每格世界尺寸
        public float maxIndent = 1.4f;       // 满压时的最大下凹深度
        public RectInt viewSize = new RectInt(0, 0, 460, 224);

        private Mesh mesh;
        private MeshFilter mf;
        private Material mat;
        private Camera cam;
        private RenderTexture rt;
        private Vector3[] verts;
        private Vector2[] uvs;
        private Texture2D heatCache;

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
            cam.fieldOfView = 42f;
            rt = new RenderTexture(viewSize.width, viewSize.height, 16);
            cam.targetTexture = rt;
            cam.transform.position = new Vector3(11f, 16f, 14f);
            cam.transform.LookAt(new Vector3(0f, -1f, 0f));
        }

        public void UpdatePressure(float[] matrix)
        {
            if (matrix == null || mesh == null) return;
            if (matrix.Length < Rows * Cols) return;

            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Cols; c++)
                {
                    int i = r * Cols + c;
                    float p = Mathf.Clamp01(matrix[i] / BedConfig.MaxPressure);
                    verts[i].y = -p * maxIndent;
                }
            }
            mesh.vertices = verts;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();

            heatCache = PressureHeatmap.Build(heatCache, matrix);
            if (mat != null && heatCache != null) mat.SetTexture("_BaseMap", heatCache);
        }
    }
}
