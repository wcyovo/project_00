using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using SmartBed.Data;
using SmartBed.Core;
using SmartBed.Visualization;

namespace SmartBed.UI
{
    /// <summary>
    /// 可视化主控制器（M1）：
    /// 轮询数据源 → 渲染压力热力图 + 压力指标 + 睡姿 + 气囊状态。
    /// 运行时自动创建整体 UI。把本组件挂到场景任意 GameObject，或直接使用 SmartBedBootstrap 自启动。
    /// </summary>
    public class SmartBedController : MonoBehaviour
    {
        [Header("数据源")]
        [Tooltip("模拟器/算法端 /stream 地址")] public string streamUrl = "http://127.0.0.1:5001/stream";
        [Tooltip("轮询频率 Hz")] public float pollRate = 5f;

        private ISmartBedDataSource source;
        private RawImage heatImage;
        private Text titleText, postureText, metricsText, airbagText, frameText;
        private Texture2D heatTex;

        private void Awake()
        {
            source = new HttpPollingDataSource();
            source.SetUrl(streamUrl);
            BuildUi();
        }

        private IEnumerator Start()
        {
            yield return new WaitForSeconds(0.4f);

            PressureMessage last = null;
            string lastErr = null;

            while (true)
            {
                lastErr = null; last = null;
                yield return source.FetchLatest(msg => last = msg, err => lastErr = err);

                if (last != null) UpdateUi(last);
                else if (!string.IsNullOrEmpty(lastErr) && frameText != null)
                    frameText.text = "连接失败: " + lastErr;

                yield return new WaitForSeconds(1f / Mathf.Max(0.1f, pollRate));
            }
        }

        // ==================== UI 更新 ====================

        private void UpdateUi(PressureMessage msg)
        {
            // 1) 压力矩阵 → 热力图
            float[] matrix = msg.pressure != null ? msg.pressure.ToArray() : null;
            matrix = PressureCalculator.EnsureLength(matrix);
            heatTex = PressureHeatmap.Build(heatTex, matrix);
            heatImage.texture = heatTex;

            // 2) 压力指标（本地计算，权威来源）
            var metrics = PressureCalculator.Compute(matrix);
            metricsText.text =
                "最大压力: " + metrics.maxPressure.ToString("F1") + " kPa\n" +
                "平均压力: " + metrics.avgPressure.ToString("F1") + " kPa\n" +
                "接触面指数: " + (metrics.contactIndex * 100f).ToString("F1") + " %";

            // 3) 睡姿
            postureText.text = msg.sleepPosture ?? "--";

            // 4) 气囊状态
            airbagText.text = BuildAirbagText(msg.airbags);

            // 5) 帧号
            frameText.text =
                "帧号: " + msg.frame +
                "   时间: " + System.DateTimeOffset.FromUnixTimeMilliseconds((long)(msg.timestamp * 1000)).LocalDateTime.ToString("HH:mm:ss");
        }

        private static string BuildAirbagText(System.Collections.Generic.List<Airbag> airbags)
        {
            if (airbags == null || airbags.Count == 0) return "气囊: --";
            var sb = new System.Text.StringBuilder("气囊状态\n");
            foreach (var a in airbags)
                sb.Append("#").Append(a.id).Append(": ").Append(a.level.ToString("F0")).Append("%\n");
            return sb.ToString().TrimEnd('\n');
        }

        // ==================== UI 构建 ====================

        private void BuildUi()
        {
            Font font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

            // Canvas
            var canvasGo = new GameObject("SmartBedCanvas");
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1600, 900);
            canvasGo.AddComponent<GraphicRaycaster>();

            // 根面板（深色背景）
            var root = UiHelper.CreatePanel(canvasGo.transform, "Root", Color.black, new Vector2(1600, 900));
            StretchFull(root);

            // 标题
            titleText = UiHelper.CreateText(root, "Title", "智能床垫实时可视化", font, 34, Color.white, TextAnchor.MiddleCenter);
            titleText.rectTransform.anchoredPosition = new Vector2(0, 410);
            titleText.rectTransform.sizeDelta = new Vector2(900, 50);

            // 热力图（背景面板 + 子级纹理）
            var heatPanel = UiHelper.CreatePanel(root, "HeatPanel", new Color(0.12f, 0.12f, 0.16f), new Vector2(480, 880));
            heatPanel.anchoredPosition = new Vector2(-500, -30);
            var texGo = new GameObject("Tex", typeof(RectTransform));
            texGo.transform.SetParent(heatPanel, false);
            heatImage = texGo.AddComponent<RawImage>();
            heatImage.color = Color.white;
            var texRt = heatImage.rectTransform;
            texRt.anchorMin = Vector2.zero;
            texRt.anchorMax = Vector2.one;
            texRt.offsetMin = new Vector2(10, 10);
            texRt.offsetMax = new Vector2(-10, -10);

            // 右侧信息栏
            var info = UiHelper.CreatePanel(root, "Info", new Color(0.08f, 0.08f, 0.11f), new Vector2(520, 780));
            info.anchoredPosition = new Vector2(480, 0);

            SetupInfoPanel(info, font);
        }

        private void SetupInfoPanel(RectTransform info, Font font)
        {
            // 睡姿
            var poseTitle = UiHelper.CreateText(info, "PoseTitle", "当前睡姿", font, 18, new Color(0.7f, 0.7f, 0.7f), TextAnchor.MiddleLeft);
            poseTitle.rectTransform.anchoredPosition = new Vector2(-230, 340);
            poseTitle.rectTransform.sizeDelta = new Vector2(200, 30);

            postureText = UiHelper.CreateText(info, "PoseValue", "--", font, 42, Color.white, TextAnchor.MiddleLeft);
            postureText.rectTransform.anchoredPosition = new Vector2(-110, 300);
            postureText.rectTransform.sizeDelta = new Vector2(300, 60);

            // 指标
            var mTitle = UiHelper.CreateText(info, "MetricsTitle", "压力指标", font, 18, new Color(0.7f, 0.7f, 0.7f), TextAnchor.MiddleLeft);
            mTitle.rectTransform.anchoredPosition = new Vector2(-230, 220);
            mTitle.rectTransform.sizeDelta = new Vector2(200, 30);

            metricsText = UiHelper.CreateText(info, "Metrics", "--", font, 24, Color.white, TextAnchor.MiddleLeft);
            metricsText.rectTransform.anchoredPosition = new Vector2(-110, 160);
            metricsText.rectTransform.sizeDelta = new Vector2(360, 140);

            // 气囊
            var aTitle = UiHelper.CreateText(info, "AirTitle", "气囊状态", font, 18, new Color(0.7f, 0.7f, 0.7f), TextAnchor.MiddleLeft);
            aTitle.rectTransform.anchoredPosition = new Vector2(-230, 40);
            aTitle.rectTransform.sizeDelta = new Vector2(200, 30);

            airbagText = UiHelper.CreateText(info, "Airbags", "--", font, 20, Color.white, TextAnchor.UpperLeft);
            airbagText.rectTransform.anchoredPosition = new Vector2(-230, 10);
            airbagText.rectTransform.sizeDelta = new Vector2(360, 160);

            // 帧号
            frameText = UiHelper.CreateText(info, "Frame", "--", font, 16, new Color(0.6f, 0.6f, 0.6f), TextAnchor.MiddleCenter);
            frameText.rectTransform.anchoredPosition = new Vector2(0, -330);
            frameText.rectTransform.sizeDelta = new Vector2(400, 30);
        }

        private static void StretchFull(RectTransform rt)
        {
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
        }

        private static class UiHelper
        {
            public static RectTransform CreatePanel(Transform parent, string name, Color color, Vector2 size)
            {
                var go = new GameObject(name, typeof(RectTransform));
                go.transform.SetParent(parent, false);
                var img = go.AddComponent<Image>();
                img.color = color;
                var rt = go.GetComponent<RectTransform>();
                rt.pivot = new Vector2(0.5f, 0.5f);
                rt.sizeDelta = size;
                return rt;
            }

            public static Text CreateText(RectTransform parent, string name, string content, Font font, int fontSize, Color color, TextAnchor anchor)
            {
                var go = new GameObject(name, typeof(RectTransform));
                go.transform.SetParent(parent, false);
                var t = go.AddComponent<Text>();
                t.font = font;
                t.text = content;
                t.fontSize = fontSize;
                t.color = color;
                t.alignment = anchor;
                t.horizontalOverflow = HorizontalWrapMode.Wrap;
                t.verticalOverflow = VerticalWrapMode.Overflow;
                return t;
            }
        }
    }
}
