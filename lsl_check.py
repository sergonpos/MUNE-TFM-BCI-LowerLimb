# lsl_check.py
# Verifica la presencia de un stream EEG en LSL (p. ej., Signal Generator),
# lista su info (name/type/fs/canales) y lee datos durante unos segundos.

from pylsl import resolve_streams, StreamInlet, LostError
import time
import numpy as np
import sys

# --- Config rápida ---
PREFER_NAME = "Signal_generator"   # pon None para ignorar el nombre y coger el primer EEG
READ_SECONDS = 5.0                 # cuánto tiempo leer muestras
PULL_TIMEOUT = 0.2                 # timeout por llamada a pull_chunk (s)
MAX_BUFLEN_S = 12                  # buffer del inlet (s)
PRINT_FIRST_SAMPLES = 3            # cuántas filas de muestra imprimir

def extract_labels(info):
    """Intenta extraer labels del XML. Devuelve list[str] o None si no hay."""
    try:
        ch = info.desc().child("channels").child("channel")
        labels = []
        while not ch.empty():
            lbl = ch.child_value("label")
            labels.append(lbl if lbl else None)
            ch = ch.next_sibling()
        if labels and any(labels):
            return labels
        return None
    except Exception:
        return None

def synth_labels(n):
    return [f"Ch{i+1}" for i in range(n)]

def pick_stream(prefer_name=None, wait_time=2.0):
    streams = resolve_streams(wait_time=wait_time)
    if not streams:
        print("No se encontraron streams LSL.")
        return None

    # 1) Intentar por nombre exacto
    if prefer_name:
        for s in streams:
            if s.name() == prefer_name:
                return s

    # 2) Si no, eligir EEG con srate>0
    eegs = [s for s in streams if s.type() == "EEG" and s.nominal_srate() > 0]
    if eegs:
        return eegs[0]

    # 3) Caer al primero si no hay EEG
    return streams[0]

def main():
    print("Buscando streams LSL...")
    info = pick_stream(PREFER_NAME, wait_time=2.0)
    if info is None:
        sys.exit(1)

    name = info.name()
    typ = info.type()
    fs = float(info.nominal_srate())
    nchan = info.channel_count()
    labels = extract_labels(info)

    print(f"name= {name}  type= {typ}  fs= {fs}  nchan= {nchan}")
    print("labels=", labels if labels is not None else None)
    if labels is None:
        labels = synth_labels(nchan)
        print("→ No había labels; usando sintéticas:", labels)

    # Abrir inlet
    inlet = StreamInlet(info, max_buflen=MAX_BUFLEN_S, processing_flags=0)
    try:
        tcorr = inlet.time_correction()  # sincronización de reloj
        print(f"time_correction={tcorr:.4f}s")
    except Exception:
        pass

    # Leer durante READ_SECONDS
    print(f"Leyendo muestras durante {READ_SECONDS:.1f}s...")
    t_start = time.perf_counter()
    last_ts = t_start
    total_samples = 0
    kept_rows = []

    try:
        while (time.perf_counter() - t_start) < READ_SECONDS:
            samples, timestamps = inlet.pull_chunk(timeout=PULL_TIMEOUT, max_samples=256)
            if timestamps:
                arr = np.asarray(samples, dtype=np.float64)
                total_samples += arr.shape[0]
                last_ts = time.perf_counter()
                if len(kept_rows) < PRINT_FIRST_SAMPLES:
                    # Guarda algunas filas para mostrar
                    need = PRINT_FIRST_SAMPLES - len(kept_rows)
                    kept_rows.extend(arr[:need].tolist())
            else:
                # Nada recibido en este tick; espera corta
                time.sleep(0.01)

        dur = max(1e-6, time.perf_counter() - t_start)
        eff_rate = total_samples / dur
        staleness = time.perf_counter() - last_ts

        print(f"\nRESULTADOS")
        print(f"- Muestras leídas: {total_samples}")
        print(f"- Duración      : {dur:.2f}s")
        print(f"- Fs nominal    : {fs:.1f} Hz")
        print(f"- Fs efectiva   : {eff_rate:.1f} Hz (aprox.)")
        print(f"- Staleness     : {staleness:.2f}s desde la última muestra")

        if kept_rows:
            print(f"\nPrimeras {len(kept_rows)} filas (forma: [n_chan={nchan}]):")
            for i, row in enumerate(kept_rows, 1):
                # imprime pocas columnas si hay muchas
                snippet = ", ".join(f"{v:.3f}" for v in row[:min(8, nchan)])
                tail = " ..." if nchan > 8 else ""
                print(f"{i:02d}: [{snippet}{tail}]")

        # Recomendación si no hay muestras
        if total_samples == 0:
            print("\n⚠️ No llegaron muestras. Revisa:")
            print("- Que el Signal Generator esté emitiendo (type='EEG', fs>0).")
            print("- Que el nombre coincida (PREFER_NAME) o pon PREFER_NAME=None.")
            print("- Firewall y multicast habilitado en tu red local.")
    except LostError:
        print("Conexión con el stream perdida.")
    finally:
        try:
            inlet.close_stream()
        except Exception:
            pass

if __name__ == "__main__":
    main()
