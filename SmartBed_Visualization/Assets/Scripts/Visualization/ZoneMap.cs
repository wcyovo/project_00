using UnityEngine;
using SmartBed.Data;

namespace SmartBed.Visualization
{
    /// <summary>
    /// 8 个气囊的网格区域映射（占位：网格坐标 44x24）。
    /// 头部区：行 0..17，左半 col 0..11（40/41/42），右半 col 12..23（64/65/66）。
    /// 脚部区：行 32..43，左半 col 0..11（12），右半 col 12..23（13）。
    /// 注：与实际气囊管路排布对齐后可再修正。
    /// </summary>
    public static class ZoneMap
    {
        public static Color ColorFor(int id)
        {
            switch (id)
            {
                case 12: return new Color(0.90f, 0.30f, 0.30f); // 脚左 - 红
                case 13: return new Color(1.00f, 0.60f, 0.20f); // 脚右 - 橙
                case 40: return new Color(0.30f, 0.80f, 0.40f);
                case 41: return new Color(0.30f, 0.70f, 0.90f);
                case 42: return new Color(0.60f, 0.50f, 0.90f);
                case 64: return new Color(0.90f, 0.85f, 0.30f);
                case 65: return new Color(0.40f, 0.75f, 0.60f);
                case 66: return new Color(0.70f, 0.60f, 0.30f);
                default: return Color.gray;
            }
        }

        /// <summary>判断网格单元 (r, c) 是否属于指定气囊的覆盖区。</summary>
        public static bool IsInZone(int id, int r, int c, int rows = BedConfig.Rows, int cols = BedConfig.Cols)
        {
            bool left = (id == 12 || id == 40 || id == 41 || id == 42);
            bool foot = (id == 12 || id == 13);

            int r0, r1, c0, c1;
            if (foot)
            {
                r0 = 32; r1 = rows - 1;
            }
            else
            {
                int band = (id == 40 || id == 64) ? 0 : (id == 41 || id == 65) ? 1 : 2;
                r0 = band * 6; r1 = r0 + 5;
            }
            if (left) { c0 = 0; c1 = 11; } else { c0 = 12; c1 = cols - 1; }
            return r >= r0 && r <= r1 && c >= c0 && c <= c1;
        }
    }
}
