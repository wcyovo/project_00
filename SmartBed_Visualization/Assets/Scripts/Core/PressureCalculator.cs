using System;
using UnityEngine;

namespace SmartBed.Core
{
    /// <summary>
    /// 压力指标计算（R3）。输入为 44x24 行优先的浮点数组（0-100 kPa）。
    /// 反映 docs/data-interface-protocol.md 第 2.4 节定义。
    /// </summary>
    public static class PressureCalculator
    {
        public static SmartBed.Data.MetricsData Compute(float[] matrix, float threshold = Data.BedConfig.ContactThreshold)
        {
            if (matrix == null || matrix.Length == 0)
                return new SmartBed.Data.MetricsData();

            float max = float.MinValue;
            double sum = 0;
            int above = 0;

            for (int i = 0; i < matrix.Length; i++)
            {
                float v = matrix[i];
                if (v > max) max = v;
                sum += v;
                if (v > threshold) above++;
            }

            return new SmartBed.Data.MetricsData
            {
                maxPressure = max,
                avgPressure = (float)(sum / matrix.Length),
                contactIndex = (float)above / matrix.Length
            };
        }

        /// <summary>保障长度，缺失时用 0 填充；超长时截断。</summary>
        public static float[] EnsureLength(float[] matrix, int expected = Data.BedConfig.CellCount)
        {
            if (matrix == null) return new float[expected];
            if (matrix.Length == expected) return matrix;
            var copy = new float[expected];
            Array.Copy(matrix, copy, Mathf.Min(matrix.Length, expected));
            return copy;
        }
    }
}
