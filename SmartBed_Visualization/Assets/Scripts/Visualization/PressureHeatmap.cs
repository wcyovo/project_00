using System.Collections.Generic;
using UnityEngine;

namespace SmartBed.Visualization
{
    /// <summary>
    /// 将 44x24 压力矩阵转成热力图纹理（R4 热力图 / R3 指标背景）。
    /// 每个传感器单元对应一个纹理像素，用点过滤保持分格清晰，再拉伸显示。
    /// </summary>
    public static class PressureHeatmap
    {
        // 颜色渐变带（暗蓝→蓝→青→绿→黄→橙→红），近似 'turbo'
        private static readonly Color[] Ramp =
        {
            new Color(0.10f, 0.10f, 0.60f), // 0
            new Color(0.10f, 0.40f, 0.90f), // 0.2
            new Color(0.20f, 0.80f, 0.90f), // 0.4
            new Color(0.30f, 0.85f, 0.40f), // 0.6
            new Color(0.95f, 0.85f, 0.20f), // 0.8
            new Color(1.00f, 0.45f, 0.10f), // 0.9
            new Color(0.90f, 0.10f, 0.10f)  // 1.0
        };

        public static Texture2D Build(Texture2D tex, float[] matrix, int rows = Data.BedConfig.Rows, int cols = Data.BedConfig.Cols, float max = Data.BedConfig.MaxPressure)
        {
            if (tex == null || tex.width != cols || tex.height != rows)
                tex = new Texture2D(cols, rows, TextureFormat.RGBA32, false);

            tex.filterMode = FilterMode.Point;
            tex.wrapMode = TextureWrapMode.Clamp;

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    int idx = r * cols + c;
                    float v = idx < matrix.Length ? matrix[idx] : 0f;
                    tex.SetPixel(c, rows - 1 - r, Sample(v / max)); // y 翻转，使第0行在上
                }
            }
            tex.Apply(false);
            return tex;
        }

        private static Color Sample(float t)
        {
            t = Mathf.Clamp01(t);
            if (Ramp.Length == 1) return Ramp[0];
            float pos = t * (Ramp.Length - 1);
            int i = Mathf.FloorToInt(pos);
            if (i >= Ramp.Length - 1) return Ramp[Ramp.Length - 1];
            return Color.Lerp(Ramp[i], Ramp[i + 1], pos - i);
        }
    }
}
