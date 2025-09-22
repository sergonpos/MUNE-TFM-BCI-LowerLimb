using System;
using System.Net;
using UnityEngine;

public class Manager : MonoBehaviour
{
    public string IP = "127.0.0.1";
    public int port = 50000;

    [Header("Referencias al juego")]
    public TrafficLightController trafficLightController;
    public ClassifierResultHandler classifierResultHandler;

    [Header("Config")]
    public int nTrials = 10;

    const int STATE_WAITING_CONNECTION = -2;
    const int STATE_WAITING_PARAMS = -1;
    const int STATE_READY = 0;

    static int state = STATE_WAITING_CONNECTION;
    static bool mustClose = false;

    private MessageInterpreter messageInterpreter = new MessageInterpreter();
    private MedusaTCPClient tcpClient;

    void Awake()
    {
        string[] arguments = Environment.GetCommandLineArgs();

        if (arguments.Length > 2 && int.TryParse(arguments[2], out int parsedPort))
        {
            IP = arguments[1];
            port = parsedPort;
        }
        else
        {
            Debug.LogWarning("⚠️ No se recibieron argumentos válidos. Usando IP/puerto por defecto.");
        }
    }

    void Start()
    {
        tcpClient = new MedusaTCPClient(this, IPAddress.Parse(IP), port);
        tcpClient.Start();
        state = STATE_WAITING_CONNECTION;
    }

    void Update()
    {
        if (state == STATE_WAITING_CONNECTION && tcpClient.isConnected())
        {
            state = STATE_WAITING_PARAMS;
            ServerMessage sm = new ServerMessage("waiting");
            tcpClient.SendMessage(sm.ToJson());
        }

        if (mustClose)
        {
            if (tcpClient.socketConnection != null)
            {
                ServerMessage sm = new ServerMessage("close");
                tcpClient.SendMessage(sm.ToJson());
            }

            mustClose = false;
            Application.Quit();
        }
    }

    public void quitApplicationFromException()
    {
        mustClose = true;
    }

    public void interpretMessage(string message)
    {
        Debug.Log("📩 Recibido: " + message);
        string eventType = messageInterpreter.decodeEventType(message);

        switch (eventType)
        {
            case "setParameters":
                // Solo responder con "ready" y quedar en espera
                onParametersReady();
                state = STATE_READY;
                break;

            case "play":
                MainThreadDispatcher.Enqueue(() =>
                {
                    trafficLightController.StartAutomaticTrials(nTrials);
                });
                break;

            case "classification_result":
            {
                float value = messageInterpreter.decodeClassificationValue(message);
                MainThreadDispatcher.Enqueue(() =>
                {
                    classifierResultHandler.ReceiveClassificationValue(value);
                });
                break;
            }


            case "stop":
                mustClose = true;
                break;

            case "exception":
                string exception = messageInterpreter.decodeException(message);
                Debug.LogError("❌ Excepción: " + exception);
                mustClose = true;
                break;

            default:
                Debug.LogWarning("⚠️ Evento no reconocido: " + eventType);
                break;
        }
    }

    void onParametersReady()
    {
        ServerMessage sm = new ServerMessage("ready");
        tcpClient.SendMessage(sm.ToJson());
        Debug.Log("📤 Enviado 'ready' a MEDUSA (en espera de 'play').");
    }

    public void SendRunFinished()
    {
        ServerMessage sm = new ServerMessage("run_finished");
        tcpClient.SendMessage(sm.ToJson());
        Debug.Log("📤 Enviado 'run_finished' a MEDUSA.");
    }
}

