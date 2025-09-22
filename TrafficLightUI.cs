using UnityEngine;
using TMPro;
using System.Collections;

public class TrafficLightUI : MonoBehaviour
{
    public TMP_Text trafficText;
    public GameObject panelBackground;

    private void OnEnable()
    {
        TrafficLightController.OnCambioColor += ActualizarTexto;
        TrafficLightController.OnSesionTerminada += MostrarFinSesion;
    }

    private void OnDisable()
    {
        TrafficLightController.OnCambioColor -= ActualizarTexto;
        TrafficLightController.OnSesionTerminada -= MostrarFinSesion;
    }

    void ActualizarTexto(string color)
    {
        switch (color)
        {
            case "Yellow":
                MostrarTexto("Get Ready...");
                break;
            case "Red":
                MostrarTexto("Stop! Rest!");
                break;
            case "Green":
                MostrarTexto("Let's Go!");
                break;
            case "Off":
                OcultarTexto();
                break;
        }
    }

    void MostrarTexto(string mensaje)
    {
        panelBackground.SetActive(true);
        trafficText.text = mensaje;
    }

    void OcultarTexto()
    {
        trafficText.text = "";
        panelBackground.SetActive(false);
    }

    void MostrarFinSesion(int correct, int total)
    {
        StopAllCoroutines();
        StartCoroutine(CoFinSesion(correct, total));
    }

    IEnumerator CoFinSesion(int correct, int total)
    {
        yield return new WaitForSeconds(1f);
        MostrarTexto($"Score: {correct}/{total}");
        // (deja el panel visible con el score hasta que cambies de escena o reinicies)
    }
}



