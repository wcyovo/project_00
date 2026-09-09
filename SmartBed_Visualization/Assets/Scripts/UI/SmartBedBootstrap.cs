using UnityEngine;

namespace SmartBed.UI
{
    /// <summary>
    /// 自启动引导：进入 Play 后自动创建 SmartBedController 并构建 UI。
    /// 因此无需手工搭场景，打开 Unity 直接按 Play 即可看到可视化。
    /// </summary>
    public static class SmartBedBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Init()
        {
            if (Object.FindObjectOfType<SmartBedController>() == null)
            {
                var go = new GameObject("SmartBed");
                go.AddComponent<SmartBedController>();
            }
        }
    }
}
