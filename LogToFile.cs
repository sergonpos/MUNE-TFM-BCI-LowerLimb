using UnityEngine;

public class LogToFile : MonoBehaviour
{
    string filename = "/unity-log.txt";
    string path = "";

    public void Log(string logString, string stackTrace, LogType type)
    {
        if (path == "")
        {
            path = Application.dataPath + filename;
            Debug.Log("📝 Log file: " + path);
        }

        try
        {
            System.IO.File.AppendAllText(path, logString + "\n");
        }
        catch { }
    }

    void OnEnable()  => Application.logMessageReceived += Log;
    void OnDisable() => Application.logMessageReceived -= Log;
}
