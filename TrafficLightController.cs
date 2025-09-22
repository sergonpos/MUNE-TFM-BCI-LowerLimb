using UnityEngine;
using System.Collections;

public class TrafficLightController : MonoBehaviour
{
    [Header("Renderers de las luces")]
    public Renderer redLight;
    public Renderer yellowLight;
    public Renderer greenLight;

    [Header("Materiales")]
    public Material OffLight;
    public Material RedLight;
    public Material YellowLight;
    public Material GreenLight;

    [Header("Audios")]
    public AudioClip beepClip;
    public AudioClip goClip;
    public AudioClip stopClip;

    [Header("Referencias externas")]
    public PlayerController playerController;
    public ClassifierResultHandler classifierResultHandler;

    [Header("Ajustes")]
    [Tooltip("Margen extra tras los 5 s de feedback para aceptar la decisión (red/latencia).")]
    public float decisionGrace = 0.15f; // 150 ms por defecto

    private AudioSource audioSource;

    public delegate void CambioColorEvent(string color);
    public static event CambioColorEvent OnCambioColor;

    public static System.Action<int,int> OnSesionTerminada;

    private int correctCount = 0;
    private string currentCue = "Off"; // "Green" o "Red" durante el CUE

    public void StartAutomaticTrials(int numTrials)
    {
        StartCoroutine(CicloSemaforo(numTrials));
    }

    IEnumerator CicloSemaforo(int numTrials)
    {
        for (int i = 0; i < numTrials; i++)
        {
            Debug.Log($"Trial {i + 1}");

            // ---- PREP 3 s (amarillo) ----
            if (playerController != null)
                playerController.StopWalking();
            if (playerController != null)
                playerController.ResetPosition();

            SetAllLightsOff();
            yellowLight.material = YellowLight;
            PlaySound(beepClip);
            currentCue = "Yellow";
            OnCambioColor?.Invoke("Yellow");
            yield return new WaitForSeconds(3f);

            // ---- CUE 2 s (rojo/verde aleatorio) ----
            int valor = Random.Range(0, 2);
            SetAllLightsOff();
            if (valor == 1)
            {
                greenLight.material = GreenLight;
                PlaySound(goClip);
                currentCue = "Green";
                OnCambioColor?.Invoke("Green");
                Debug.Log("Verde");
            }
            else
            {
                redLight.material = RedLight;
                PlaySound(stopClip);
                currentCue = "Red";
                OnCambioColor?.Invoke("Red");
                Debug.Log("Rojo");
            }
            yield return new WaitForSeconds(2f);

            // ---- FEEDBACK 5 s ----
            classifierResultHandler.ResetVentanaDecision();
            classifierResultHandler.esperandoClasificacion = true;

            yield return new WaitForSeconds(5f + Mathf.Max(0f, decisionGrace));

            classifierResultHandler.esperandoClasificacion = false;

            // ---- SCORING en base al último bit visto dentro de la ventana ----
            int expectedBit = (currentCue == "Green") ? 1 : 0;
            int? received = classifierResultHandler.ultimoBitRecibido;

            Debug.Log($"[TLC] recibido={(received.HasValue ? received.Value.ToString() : "null")} / esperado={expectedBit}");

            if (received.HasValue && received.Value == expectedBit)
            {
                correctCount++;
                Debug.Log($"ACIERTO (recibido={received.Value}, esperado={expectedBit}). Score parcial: {correctCount}/{i+1}");
            }
            else
            {
                Debug.Log($"FALLO (recibido={(received.HasValue?received.Value.ToString():"null")}, esperado={expectedBit}). Score parcial: {correctCount}/{i+1}");
            }

            // ---- REST 5 s (luces OFF) ----
            SetAllLightsOff();
            currentCue = "Off";
            OnCambioColor?.Invoke("Off");
            yield return new WaitForSeconds(5f);
        }

        // Score display
        OnSesionTerminada?.Invoke(correctCount, numTrials);

        yield return new WaitForSeconds(5f);

        // Notificar a MEDUSA
        Manager manager = FindFirstObjectByType<Manager>();
        if (manager != null)
            manager.SendRunFinished();
    }

    void SetAllLightsOff()
    {
        redLight.material = OffLight;
        yellowLight.material = OffLight;
        greenLight.material = OffLight;
    }

    void PlaySound(AudioClip clip)
    {
        if (clip != null && audioSource == null)
            audioSource = GetComponent<AudioSource>();

        if (clip != null && audioSource != null)
        {
            audioSource.Stop();
            audioSource.PlayOneShot(clip);
        }
    }
}


