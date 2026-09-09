using System.Collections.Generic;
using UnityEngine;
using SmartBed.Data;

namespace SmartBed.Visualization
{
    /// <summary>
    /// 气囊调节策略（可视化端自推，R1/R4）。
    /// 依据压力分布推导 8 个气囊的充气量：某个气囊覆盖区的平均压力越高，
    /// 该气囊越倾向“放气减压”，以重新分布压力（support）。level = 100 - 区域平均压力占比%。
    /// 若外部（组员算法/硬件）提供了真实气囊状态，可替换掉本推导结果。
    /// </summary>
    public static class AirbagStrategy
    {
        private static readonly int[] Ids = { 12, 13, 40, 41, 42, 64, 65, 66 };

        public static List<Airbag> Derive(float[] matrix, int rows = BedConfig.Rows, int cols = BedConfig.Cols)
        {
            var result = new List<Airbag>();
            if (matrix == null || matrix.Length < rows * cols) return result;

            foreach (var id in Ids)
            {
                float sum = 0f;
                int count = 0;
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        if (ZoneMap.IsInZone(id, r, c))
                        {
                            sum += matrix[r * cols + c];
                            count++;
                        }
                    }
                }
                float zoneAvg = count > 0 ? sum / count : 0f;
                float level = 100f * (1f - Mathf.Clamp01(zoneAvg / BedConfig.MaxPressure));
                result.Add(new Airbag { id = id, level = Mathf.Round(level) });
            }
            return result;
        }
    }
}
