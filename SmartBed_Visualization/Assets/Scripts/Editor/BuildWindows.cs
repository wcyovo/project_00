using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

/// <summary>
/// Windows 一键打包脚本。
/// 菜单：Build → Windows Build；也支持命令行：-executeMethod BuildWindows.Build
/// 输出：Build/Windows/SmartBedVisualization.exe
/// </summary>
public static class BuildWindows
{
    [MenuItem("Build/Windows Build")]
    public static void Build()
    {
        var options = new BuildPlayerOptions
        {
            scenes = new[] { "Assets/Scenes/SampleScene.unity" },
            locationPathName = "Build/Windows/SmartBedVisualization.exe",
            target = BuildTarget.StandaloneWindows64,
            options = BuildOptions.None
        };

        BuildReport report = BuildPipeline.BuildPlayer(options);
        if (report.summary.result == BuildResult.Succeeded)
        {
            Debug.Log($"[Build] 构建成功！输出：{report.summary.outputPath} ({report.summary.totalSize} 字节)");
        }
        else
        {
            Debug.LogError($"[Build] 构建失败：{report.summary.result}");
        }
    }
}
