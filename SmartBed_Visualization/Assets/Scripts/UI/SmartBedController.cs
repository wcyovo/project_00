using System.Collections;
using System.Collections.Generic;
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
        private RawImage overlayImage;
        private RawImage airbagChart;
        private RawImage sensorChart;
        private RawImage bed3dImage;
        private Bed3DView bed3d;
        private Text titleText, userText, postureText, metricsText, airbagText, frameText;
        private Texture2D heatTex;
        private Texture2D overlayTex;
        private Texture2D airbagChartTex;
        private Texture2D sensorChartTex;

        // 中部图表尺寸（与 AddChartBlock 的 panel 尺寸对应）
        private const int ChartW = 440;
        private const int ChartH = 224;

        // R4 动态曲线：滚动历史
        [Tooltip("曲线最大历史帧数")] public int maxHistory = 90;
        private readonly List<float[]> levelHistory = new List<float[]>();   // 每条 = 8 气囊
        private readonly List<float[]> pressHistory = new List<float[]>();   // 每条 = k 个传感点
        private static readonly int[] AirbagIds = { 12, 13, 40, 41, 42, 64, 65, 66 };
        // 供曲线展示的代表性传感点：肩 / 腰 / 臀（沿纵向中线的网格索引 = row*24+col）
        private static readonly int[] SensorIndices = { 6 * 24 + 12, 18 * 24 + 12, 30 * 24 + 12 };

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

            // 1c) 3D 床垫形变
            if (bed3d != null) bed3d.UpdatePressure(matrix);

            // 1b) 支撑叠加层（身体部位框 + 气囊充气着色）—— 气囊量由可视化自推
            List<Airbag> airbags = AirbagStrategy.Derive(matrix);
            overlayTex = SupportOverlay.Build(overlayTex, airbags, msg.bodyRegions);
            overlayImage.texture = overlayTex;

            // 2) 压力指标（本地计算，权威来源）
            var metrics = PressureCalculator.Compute(matrix);
            metricsText.text =
                "最大压力: " + metrics.maxPressure.ToString("F1") + " kPa\n" +
                "平均压力: " + metrics.avgPressure.ToString("F1") + " kPa\n" +
                "接触面指数: " + (metrics.contactIndex * 100f).ToString("F1") + " %";

            // 3) 睡姿
            postureText.text = msg.sleepPosture ?? "--";

            // 4) 当前用户（来自用户识别模块，可空）
            userText.text = "当前用户: " + (string.IsNullOrEmpty(msg.currentUser) ? "--" : msg.currentUser);

            // 5) 气囊状态（自推策略）
            airbagText.text = BuildAirbagText(airbags);

            // 6) 帧号
            frameText.text =
                "帧号: " + msg.frame +
                "   时间: " + System.DateTimeOffset.FromUnixTimeMilliseconds((long)(msg.timestamp * 1000)).LocalDateTime.ToString("HH:mm:ss");

            // 7) R4 动态曲线：追加历史并重绘
            AppendHistory(airbags, matrix);
            DrawCharts();
        }

        private void AppendHistory(List<Airbag> airbags, float[] matrix)
        {
            // 气囊量（自推）
            float[] levels = new float[AirbagIds.Length];
            for (int i = 0; i < AirbagIds.Length; i++)
            {
                for (int j = 0; j < airbags.Count; j++)
                {
                    if (airbags[j].id == AirbagIds[i]) { levels[i] = airbags[j].level; break; }
                }
            }
            levelHistory.Add(levels);

            // 传感点压力
            float[] pressures = new float[SensorIndices.Length];
            for (int j = 0; j < SensorIndices.Length; j++)
                pressures[j] = (SensorIndices[j] >= 0 && SensorIndices[j] < matrix.Length) ? matrix[SensorIndices[j]] : 0f;
            pressHistory.Add(pressures);

            if (levelHistory.Count > maxHistory) levelHistory.RemoveAt(0);
            if (pressHistory.Count > maxHistory) pressHistory.RemoveAt(0);
        }

        private void DrawCharts()
        {
            // 气囊曲线
            var airColors = new Color[AirbagIds.Length];
            for (int i = 0; i < AirbagIds.Length; i++) airColors[i] = ZoneMap.ColorFor(AirbagIds[i]);
            airbagChartTex = ChartRenderer.Draw(airbagChartTex, ToSeries(levelHistory, AirbagIds.Length), airColors, ChartW, ChartH, 100f);
            airbagChart.texture = airbagChartTex;

            // 传感点压力曲线
            var sensorColors = new Color[] { Color.white, Color.magenta, Color.cyan };
            sensorChartTex = ChartRenderer.Draw(sensorChartTex, ToSeries(pressHistory, SensorIndices.Length), sensorColors, ChartW, ChartH, 100f);
            sensorChart.texture = sensorChartTex;
        }

        // 把历史列表转成 ChartRenderer 需要的“按序列”的数组集
        private static List<float[]> ToSeries(List<float[]> history, int seriesCount)
        {
            var result = new List<float[]>();
            int n = history.Count;
            if (n == 0) return result;
            for (int c = 0; c < seriesCount; c++)
            {
                var arr = new float[n];
                for (int k = 0; k < n; k++) arr[k] = history[k][c];
                result.Add(arr);
            }
            return result;
        }

        private static string BuildAirbagText(System.Collections.Generic.List<Airbag> airbags)
        {
            if (airbags == null || airbags.Count == 0) return "气囊: --";
            var sb = new System.Text.StringBuilder();
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

            // 标题（架构：大屏可视化系统）
            titleText = UiHelper.CreateText(root, "Title", "大屏可视化系统", font, 34, Color.white, TextAnchor.MiddleCenter);
            titleText.rectTransform.anchoredPosition = new Vector2(0, 410);
            titleText.rectTransform.sizeDelta = new Vector2(900, 50);

            var subtitle = UiHelper.CreateText(root, "SubTitle", "智能床垫实时数据可视化", font, 18, new Color(0.8f, 0.8f, 0.8f), TextAnchor.MiddleCenter);
            subtitle.rectTransform.anchoredPosition = new Vector2(0, 382);
            subtitle.rectTransform.sizeDelta = new Vector2(900, 26);

            // 热力图（背景面板 + 子级纹理）
            var heatPanel = UiHelper.CreatePanel(root, "HeatPanel", new Color(0.12f, 0.12f, 0.16f), new Vector2(480, 880));
            heatPanel.anchoredPosition = new Vector2(-500, -30);

            // 分区标签：床垫受压数据
            var heatTitle = UiHelper.CreateText(root, "HeatTitle", "床垫受压数据", font, 20, new Color(0.98f, 0.85f, 0.4f), TextAnchor.MiddleCenter);
            heatTitle.rectTransform.anchoredPosition = new Vector2(-640, 432);
            heatTitle.rectTransform.sizeDelta = new Vector2(240, 30);
            var texGo = new GameObject("Tex", typeof(RectTransform));
            texGo.transform.SetParent(heatPanel, false);
            heatImage = texGo.AddComponent<RawImage>();
            heatImage.color = Color.white;
            var texRt = heatImage.rectTransform;
            texRt.anchorMin = Vector2.zero;
            texRt.anchorMax = Vector2.one;
            texRt.offsetMin = new Vector2(10, 10);
            texRt.offsetMax = new Vector2(-10, -10);

            // 支撑叠加层（与热力图同区域，显示身体部位框 + 气囊充气着色）
            var ovGo = new GameObject("Overlay", typeof(RectTransform));
            ovGo.transform.SetParent(heatPanel, false);
            overlayImage = ovGo.AddComponent<RawImage>();
            overlayImage.color = Color.white;
            overlayImage.raycastTarget = false;
            var ovRt = overlayImage.rectTransform;
            ovRt.anchorMin = Vector2.zero;
            ovRt.anchorMax = Vector2.one;
            ovRt.offsetMin = new Vector2(10, 10);
            ovRt.offsetMax = new Vector2(-10, -10);

            // 右侧信息栏
            var info = UiHelper.CreatePanel(root, "Info", new Color(0.08f, 0.08f, 0.11f), new Vector2(520, 780));
            info.anchoredPosition = new Vector2(480, 0);

            SetupInfoPanel(info, font);

            // 中部列：3D 床垫（上） + 气囊曲线（中） + 传感点曲线（下）
            var size = new Vector2(460, 240);
            bed3dImage = AddChartBlock(root, font, "Bed3D", "3D 床垫支撑形变", new Vector2(-15, 250), size);
            var bedGo = new GameObject("Bed3D");
            bed3d = bedGo.AddComponent<Bed3DView>();
            bed3dImage.texture = bed3d.RenderTexture;

            airbagChart = AddChartBlock(root, font, "AirbagChart", "气囊充气量曲线 (0-100%)", new Vector2(-15, 10), size);
            sensorChart = AddChartBlock(root, font, "SensorChart", "传感点压力曲线 (0-100 kPa)", new Vector2(-15, -230), size);
        }

        private RawImage AddChartBlock(RectTransform root, Font font, string name, string title, Vector2 center, Vector2 size)
        {
            var panel = UiHelper.CreatePanel(root, name, new Color(0.06f, 0.06f, 0.10f), size);
            panel.anchoredPosition = center;

            var t = UiHelper.CreateText(panel, name + "Title", title, font, 16, new Color(0.8f, 0.8f, 0.8f), TextAnchor.MiddleCenter);
            t.rectTransform.anchoredPosition = new Vector2(0, size.y * 0.5f - 15);
            t.rectTransform.sizeDelta = new Vector2(size.x - 20, 30);

            var go = new GameObject(name + "Img", typeof(RectTransform));
            go.transform.SetParent(panel, false);
            var img = go.AddComponent<RawImage>();
            img.color = Color.white;
            img.raycastTarget = false;
            var rt = img.rectTransform;
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(8, 8);
            rt.offsetMax = new Vector2(-8, -8);
            return img;
        }

        private void SetupInfoPanel(RectTransform info, Font font)
        {
            // 当前用户（顶部）
            userText = UiHelper.CreateText(info, "CurrentUser", "当前用户: --", font, 20, new Color(0.95f, 0.85f, 0.4f), TextAnchor.MiddleLeft);
            userText.rectTransform.anchoredPosition = new Vector2(-110, 372);
            userText.rectTransform.sizeDelta = new Vector2(360, 30);

            // 睡姿（架构：用户睡姿）
            var poseTitle = UiHelper.CreateText(info, "PoseTitle", "用户睡姿", font, 18, new Color(0.7f, 0.7f, 0.7f), TextAnchor.MiddleLeft);
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

            // 气囊（架构：气垫状态）
            var aTitle = UiHelper.CreateText(info, "AirTitle", "气垫状态", font, 18, new Color(0.7f, 0.7f, 0.7f), TextAnchor.MiddleLeft);
            aTitle.rectTransform.anchoredPosition = new Vector2(-240, 120);
            aTitle.rectTransform.sizeDelta = new Vector2(220, 30);

            airbagText = UiHelper.CreateText(info, "Airbags", "--", font, 18, Color.white, TextAnchor.UpperLeft);
            airbagText.rectTransform.anchoredPosition = new Vector2(-240, 90);
            airbagText.rectTransform.sizeDelta = new Vector2(400, 240);
            airbagText.horizontalOverflow = HorizontalWrapMode.Overflow;

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
