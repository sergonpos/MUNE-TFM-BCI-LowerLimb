# tcp_binary_feeder_after_feedback.py
# Envía 0/1 al FINAL de los 5 s de feedback (justo al entrar en REST).

import socket, json, time, itertools

HOST = "127.0.0.1"
PORT = 50000

# ---- Timings (s) ----
PREP = 3.0
CUE = 2.0
FEEDBACK = 5.0
REST = 5.0
N_TRIALS = 10

# Patrón de bits a enviar (cámbialo como quieras)
VALUES = [1,0,1,1,0,0,1,0,1,0]  # o: list(itertools.islice(itertools.cycle([1,0]), N_TRIALS))

JSON_HEADER_DEFAULTS = {
    "byteorder": "little",
    "content-type": "text/json",
    "content-encoding": "utf-8",
}
PROTO_HEADER_LEN_BYTES = 2
PROTO_HEADER_BYTEORDER = "big"

def send_event(sock, event_type: str, **extra):
    payload = {"event_type": event_type}
    payload.update(extra)
    json_bytes = json.dumps(payload).encode("utf-8")
    hdr = dict(JSON_HEADER_DEFAULTS)
    hdr["content-length"] = len(json_bytes)
    hdr_bytes = json.dumps(hdr).encode("utf-8")
    pfx = len(hdr_bytes).to_bytes(PROTO_HEADER_LEN_BYTES, PROTO_HEADER_BYTEORDER)
    sock.sendall(pfx + hdr_bytes + json_bytes)

def main():
    print(f"🖧 Esperando conexión Unity en {HOST}:{PORT} ...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)
        conn, addr = srv.accept()
        with conn:
            print(f"✅ Unity conectado: {addr}")

            # Leer (opcional) el 'waiting' inicial si llega
            try:
                conn.settimeout(1.0)
                _ = conn.recv(4096)
                print("📩 Mensaje inicial de Unity recibido.")
            except Exception:
                pass
            finally:
                conn.settimeout(None)

            # setParameters (+ n_trials informativo)
            settings_payload = {
                "timings": {
                    "o_trial_cue": int(PREP*1000),
                    "o_trial_start_feedback": int((PREP+CUE)*1000),
                    "o_trial_final_feedback": int((PREP+CUE+FEEDBACK)*1000),
                    "t_trial_rest": int(REST*1000),
                    "n_trials": N_TRIALS,
                }
            }
            send_event(conn, "setParameters", payload=settings_payload)
            print("📤 setParameters enviado.")

            # play
            time.sleep(0.2)
            send_event(conn, "play")
            print("▶️  play enviado. Empiezan los trials.")

            for i, val in enumerate(VALUES[:N_TRIALS], start=1):
                t0 = time.perf_counter()
                print(f"\n[Trial {i}/{N_TRIALS}]")

                # Esperar hasta el FINAL del feedback
                decision_at = PREP + CUE + FEEDBACK
                while (time.perf_counter() - t0) < decision_at:
                    time.sleep(0.002)

                # Enviar 0/1 justo al entrar en REST
                send_event(conn, "classification_result", value=int(val))
                print(f"📤 classification_result (post-FB) = {val}")

                # Esperar resto del trial (REST)
                while (time.perf_counter() - t0) < (PREP + CUE + FEEDBACK + REST):
                    time.sleep(0.002)

            # stop
            time.sleep(0.3)
            send_event(conn, "stop")
            print("\n🛑 stop enviado. Secuencia terminada.")

if __name__ == "__main__":
    main()
