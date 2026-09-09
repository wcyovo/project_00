using System.Collections.Generic;
using UnityEngine;

namespace SmartBed.Visualization
{
    /// <summary>
    /// 简易折线图渲染器（R4）：在纹理上绘制多条等长序列的折线 + 网格 + 边框。
    /// 用于展示气囊充气量、指定传感点压力的随时间变化。
    /// </summary>
    public static class ChartRenderer
    {
        public static Texture2D Draw(Texture2D tex, IReadOnlyList<float[]> series, IReadOnlyList<Color> colors,
            int width, int height, float maxValue = 100f, int pad = 10)
        {
            if (tex == null || tex.width != width || tex.height != height)
                tex = new Texture2D(width, height, TextureFormat.RGBA32, false);
            tex.filterMode = FilterMode.Bilinear;
            tex.wrapMode = TextureWrapMode.Clamp;

            // 背景
            Color bg = new Color(0.05f, 0.05f, 0.08f);
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    tex.SetPixel(x, y, bg);

            int dataLen = series.Count > 0 ? series[0].Length : 0;
            int innerW = width - pad * 2;
            int innerH = height - pad * 2;
            if (dataLen < 2) { tex.Apply(false); return tex; }

            // 网格
            Color grid = new Color(0.22f, 0.22f, 0.28f);
            for (int gy = 0; gy <= 4; gy++)
            {
                int py = pad + innerH * gy / 4;
                for (int x = 0; x < width; x++) tex.SetPixel(x, py, grid);
            }
            for (int gx = 0; gx <= 5; gx++)
            {
                int px = pad + innerW * gx / 5;
                for (int y = 0; y < height; y++) tex.SetPixel(px, y, grid);
            }

            // 曲线
            for (int s = 0; s < series.Count; s++)
            {
                float[] data = series[s];
                if (data == null || data.Length < 2) continue;
                Color col = s < colors.Count ? colors[s] : Color.white;
                for (int i = 1; i < data.Length; i++)
                {
                    float v0 = Mathf.Clamp(data[i - 1], 0f, maxValue) / maxValue;
                    float v1 = Mathf.Clamp(data[i], 0f, maxValue) / maxValue;
                    float x0 = pad + (float)(i - 1) / (dataLen - 1) * innerW;
                    float y0 = pad + (1f - v0) * innerH;
                    float x1 = pad + (float)(i) / (dataLen - 1) * innerW;
                    float y1 = pad + (1f - v1) * innerH;
                    DrawLine(tex, Mathf.RoundToInt(x0), Mathf.RoundToInt(y0), Mathf.RoundToInt(x1), Mathf.RoundToInt(y1), col);
                }
            }

            tex.Apply(false);
            return tex;
        }

        private static void DrawLine(Texture2D tex, int x0, int y0, int x1, int y1, Color c)
        {
            int dx = Mathf.Abs(x1 - x0), dy = Mathf.Abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
            int err = dx - dy;
            while (true)
            {
                if (x0 >= 0 && x0 < tex.width && y0 >= 0 && y0 < tex.height) tex.SetPixel(x0, y0, c);
                if (x0 == x1 && y0 == y1) break;
                int e2 = 2 * err;
                if (e2 > -dy) { err -= dy; x0 += sx; }
                if (e2 < dx) { err += dx; y0 += sy; }
            }
        }
    }
}
