using System.Collections.Generic;
using UnityEngine;
using SmartBed.Data;

namespace SmartBed.Visualization
{
    /// <summary>
    /// 支撑叠加层（R1）：在压力热力图之上叠加两层信息 ——
    /// 1) 气囊覆盖区按当前充气量着色（透明度随 level 变化，体现“调节支撑”）；
    /// 2) 身体部位框（肩/背/腰/臀/大腿）边框。
    /// 生成一张与热力图同尺寸的 44x24 透明叠加纹理。
    /// </summary>
    public static class SupportOverlay
    {
        public static Texture2D Build(Texture2D tex, List<Airbag> airbags, List<BodyRegion> regions,
            int rows = BedConfig.Rows, int cols = BedConfig.Cols)
        {
            if (tex == null || tex.width != cols || tex.height != rows)
                tex = new Texture2D(cols, rows, TextureFormat.RGBA32, false);

            tex.filterMode = FilterMode.Point;
            tex.wrapMode = TextureWrapMode.Clamp;

            // 清为透明
            Color clear = new Color(0f, 0f, 0f, 0f);
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    tex.SetPixel(c, rows - 1 - r, clear);

            // 1) 气囊区着色（透明度 = 充气量）
            if (airbags != null)
            {
                foreach (var a in airbags)
                {
                    Color col = ZoneMap.ColorFor(a.id);
                    col.a = Mathf.Clamp01(a.level / 100f) * 0.55f;
                    for (int r = 0; r < rows; r++)
                        for (int c = 0; c < cols; c++)
                            if (ZoneMap.IsInZone(a.id, r, c))
                                tex.SetPixel(c, rows - 1 - r, col);
                }
            }

            // 2) 身体部位框边框
            if (regions != null)
            {
                foreach (var reg in regions)
                	DrawBorder(tex, reg.part, reg.x1, reg.y1, reg.x2, reg.y2, rows, cols);
            }

            tex.Apply(false);
            return tex;
        }

        private static void DrawBorder(Texture2D tex, string part, float fx1, float fy1, float fx2, float fy2, int rows, int cols)
        {
            int x1 = Mathf.Clamp(Mathf.RoundToInt(fx1), 0, cols - 1);
            int x2 = Mathf.Clamp(Mathf.RoundToInt(fx2), 0, cols - 1);
            int y1 = Mathf.Clamp(Mathf.RoundToInt(fy1), 0, rows - 1);
            int y2 = Mathf.Clamp(Mathf.RoundToInt(fy2), 0, rows - 1);
            if (x2 < x1) { int t = x1; x1 = x2; x2 = t; }
            if (y2 < y1) { int t = y1; y1 = y2; y2 = t; }

            Color c = ColorForPart(part);
            for (int x = x1; x <= x2; x++) { Set(tex, x, y1, c); Set(tex, x, y2, c); }
            for (int y = y1; y <= y2; y++) { Set(tex, x1, y, c); Set(tex, x2, y, c); }
        }

        private static Color ColorForPart(string part)
        {
            switch (part)
            {
                case "肩": return new Color(0.95f, 0.20f, 0.20f, 1f);
                case "背": return new Color(1.00f, 0.55f, 0.10f, 1f);
                case "腰": return new Color(0.95f, 0.90f, 0.20f, 1f);
                case "臀": return new Color(0.20f, 0.85f, 0.30f, 1f);
                case "大腿": return new Color(0.20f, 0.50f, 1.00f, 1f);
                default: return Color.white;
            }
        }

        private static void Set(Texture2D tex, int x, int y, Color c)
        {
            tex.SetPixel(x, tex.height - 1 - y, c); // 与热力图一致的 y 翻转
        }
    }
}
