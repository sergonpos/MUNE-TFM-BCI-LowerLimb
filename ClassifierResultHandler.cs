using UnityEngine;

public class ClassifierResultHandler : MonoBehaviour
{
    public PlayerController playerController;
    public bool esperandoClasificacion = false;
    public int? ultimoBitRecibido = null;

    public void ResetVentanaDecision()
    {
        ultimoBitRecibido = null;
    }

    public void ReceiveClassificationValue(float value)
    {
        int bit = Mathf.RoundToInt(value);

        Debug.Log($"Recibido bit={bit} (esperando={esperandoClasificacion}, inst={GetInstanceID()})");

        if (!esperandoClasificacion) return;

        ultimoBitRecibido = bit;

        if (bit == 1)
        {
            if (playerController) playerController.StartWalking();
            Debug.Log("MOVE (continuous walking)");
        }
        else
        {
            if (playerController) playerController.StopWalking();
            Debug.Log("STOP (binary=0)");
        }
    }
}

