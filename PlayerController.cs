using UnityEngine;

public class PlayerController : MonoBehaviour
{
    [Header("Movement")]
    public float moveDistance = 1.5f;   // step length (para MoveForward)
    public float moveSpeed = 2f;        // units/sec (sirve para paso y continuo)

    private Animator animator;
    private Rigidbody rb;

    // Estado “paso”
    private Vector3 targetPosition;
    private bool isStepping = false;

    // Estado “caminar continuo”
    private bool isWalkingContinuous = false;

    // Pose inicial para reset
    private Vector3 initialPosition;
    private Quaternion initialRotation;

    void Awake()
    {
        animator = GetComponent<Animator>();
        rb = GetComponent<Rigidbody>();

        initialPosition = transform.position;
        initialRotation = transform.rotation;

        targetPosition = transform.position;
    }

    void Update()
    {
        // 1) Paso discreto (hasta target)
        if (isStepping)
        {
            SetAnim(true);

            Vector3 currentPos = rb ? rb.position : transform.position;
            Vector3 newPos = Vector3.MoveTowards(currentPos, targetPosition, moveSpeed * Time.deltaTime);

            if (rb) rb.MovePosition(newPos);
            else    transform.position = newPos;

            if (Vector3.Distance(newPos, targetPosition) < 0.01f)
            {
                isStepping = false;
                StopPhysics();
                SetAnim(false);
            }
            return; // prioridad al modo “paso” si está activo
        }

        // 2) Caminar continuo
        if (isWalkingContinuous)
        {
            SetAnim(true);

            Vector3 currentPos = rb ? rb.position : transform.position;
            Vector3 newPos = currentPos + transform.forward * (moveSpeed * Time.deltaTime);

            if (rb) rb.MovePosition(newPos);
            else    transform.position = newPos;

            return;
        }

        // 3) Quieto
        SetAnim(false);
    }

    // ---- API pública ----

    // Paso único (compatible con lo que ya tenías)
    public void MoveForward()
    {
        if (isStepping) return;
        Vector3 basis = rb ? rb.position : transform.position;
        targetPosition = basis + transform.forward * moveDistance;
        isStepping = true;
        isWalkingContinuous = false;
    }

    // Empezar a caminar de forma continua (hasta que alguien llame StopWalking/ResetPosition)
    public void StartWalking()
    {
        isStepping = false;
        isWalkingContinuous = true;
    }

    // Parar el caminar continuo (se queda donde esté)
    public void StopWalking()
    {
        isWalkingContinuous = false;
        StopPhysics();
        SetAnim(false);
    }

    // Reset duro a la pose inicial (llámalo al poner AMARILLO)
    public void ResetPosition()
    {
        isStepping = false;
        isWalkingContinuous = false;

        StopPhysics();

        if (rb) { rb.position = initialPosition; rb.MovePosition(initialPosition); }
        else    { transform.position = initialPosition; }

        transform.rotation = initialRotation;
        targetPosition = initialPosition;
        SetAnim(false);
    }

    // ---- utilidades ----
    void StopPhysics()
    {
        if (rb)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
    }

    void SetAnim(bool walking)
    {
        if (animator) animator.SetBool("IsWalking", walking);
    }
}






