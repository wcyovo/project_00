using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

namespace SmartBed.Data
{
    /// <summary>床垫固定规格（对齐 docs/mattress-spec.md）。</summary>
    public static class BedConfig
    {
        public const int Rows = 44;                 // 行数
        public const int Cols = 24;                 // 列数
        public const int CellCount = Rows * Cols;   // 1056
        public const float MaxPressure = 100f;      // 归一化上限 kPa
        public const float ContactThreshold = 10f;  // 接触面指数阈值 kPa
    }

    // ---- 协议数据模型（字段名与 docs/data-interface-protocol.md 对齐）----

    [Serializable]
    public class BodyRegion
    {
        public string part;
        public float x1, y1, x2, y2;   // 网格坐标 x∈0..24, y∈0..44
    }

    [Serializable]
    public class Airbag
    {
        public int id;
        public float level;            // 充气量 0-100
    }

    [Serializable]
    public class MetricsData
    {
        public float maxPressure;
        public float avgPressure;
        public float contactIndex;
    }

    [Serializable]
    public class PressureMessage
    {
        public double timestamp;
        public int frame;
        public List<float> pressure;   // 1056 个浮点，行优先
        public string sleepPosture;    // 仰卧/俯卧/左侧卧/右侧卧
        public int sleepPoseIndex;
        public List<BodyRegion> bodyRegions;
        public List<Airbag> airbags;
        public MetricsData metrics;
    }

    // ---- 数据源接口 ----

    public interface ISmartBedDataSource
    {
        void SetUrl(string url);
        IEnumerator FetchLatest(Action<PressureMessage> onOk, Action<string> onError);
    }

    /// <summary>HTTP 轮询数据源（GET 单帧 JSON）。</summary>
    public class HttpPollingDataSource : ISmartBedDataSource
    {
        private string url = "http://127.0.0.1:5001/stream";

        public void SetUrl(string u) => url = u;

        public IEnumerator FetchLatest(Action<PressureMessage> onOk, Action<string> onError)
        {
            using (var req = UnityWebRequest.Get(url))
            {
                req.timeout = 5;
                yield return req.SendWebRequest();

                if (req.result == UnityWebRequest.Result.Success)
                {
                    try
                    {
                        var msg = JsonUtility.FromJson<PressureMessage>(req.downloadHandler.text);
                        onOk?.Invoke(msg);
                    }
                    catch (Exception e)
                    {
                        onError?.Invoke("JSON 解析失败: " + e.Message);
                    }
                }
                else
                {
                    onError?.Invoke(req.error);
                }
            }
        }
    }
}
