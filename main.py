# MEDUSA-compatible, no app_controller / no Qt menus.
# LSL EEG → preprocessing → MI & ATT (5 s epochs) → gating → ONE binary (0/1) per trial
# TCP out: Unity (framed JSON) + MATLAB (text line)


import os, sys, logging, time, threading
def make_logger(name="App"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(h)
    return logger

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Importa el shim de forma robusta
try:
    from shim_lsl import LSLWorkerShim
except Exception as _imp_err:
    # fallback por ruta absoluta (por si MEDUSA cambia el cwd)
    import importlib.util
    shim_path = os.path.join(HERE, "shim_lsl.py")
    spec = importlib.util.spec_from_file_location("shim_lsl", shim_path)
    shim_lsl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim_lsl)
    LSLWorkerShim = shim_lsl.LSLWorkerShim


import json, socket
from collections import deque
from typing import Optional, Dict
import os, subprocess
import numpy as np
from joblib import load
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import iirnotch, butter, filtfilt
from gui import gui_utils
from medusa import emg, nirs
from medusa.bci.mi_paradigms import *

# --- ensure local imports work even if MEDUSA changes CWD ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from settings import Settings

# --- MEDUSA core ---
import resources, exceptions
import constants as mds_constants

# =============== DSP utils ===============
def demean(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=0, keepdims=True)

def apply_notch(x: np.ndarray, fs: float, f0: float = 50.0, Q: float = 35.0) -> np.ndarray:
    b, a = iirnotch(w0=f0/(fs/2), Q=Q)
    return filtfilt(b, a, x, axis=0)

def bandpass(x: np.ndarray, fs: float, fmin: float, fmax: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, [fmin/(fs/2), fmax/(fs/2)], btype='band')
    return filtfilt(b, a, x, axis=0)

def apply_car(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=1, keepdims=True)


# =============== ATT features (frontal) ===============
FRONTAL = ['F3', 'Fz', 'F4']
BANDS_ATT = {'theta': (4, 7), 'alpha': (8, 12), 'betaL': (13, 20)}  # ajusta si tu training difiere

def _welch_logpow(sig: np.ndarray, fs: float, fmin: float, fmax: float, win_s: float = 2.0, overlap: float = 0.5) -> float:
    from scipy.signal import welch
    nperseg = int(round(win_s * fs))
    noverlap = int(round(nperseg * overlap))
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    m = (f >= fmin) & (f <= fmax)
    return float(np.log(np.trapz(Pxx[m], f[m]) + 1e-12))

def extract_attention_features(epoch_2d: np.ndarray, fs: float, ch_names: list) -> np.ndarray:
    ep = apply_car(epoch_2d)
    idx = {ch: i for i, ch in enumerate(ch_names)}
    have = [ch for ch in FRONTAL if ch in idx] or ch_names[:min(3, len(ch_names))]
    feats = []
    for ch in have:
        s = ep[:, idx[ch]]
        th = _welch_logpow(s, fs, *BANDS_ATT['theta'])
        al = _welch_logpow(s, fs, *BANDS_ATT['alpha'])
        be = _welch_logpow(s, fs, *BANDS_ATT['betaL'])
        th_al = th - al
        th_be = th - be
        ei = be - np.log(np.exp(al) + np.exp(th) + 1e-12)
        feats += [th, al, be, th_al, th_be, ei]

    s_avg = np.mean(ep[:, [idx[ch] for ch in have]], axis=1)
    th = _welch_logpow(s_avg, fs, *BANDS_ATT['theta'])
    al = _welch_logpow(s_avg, fs, *BANDS_ATT['alpha'])
    be = _welch_logpow(s_avg, fs, *BANDS_ATT['betaL'])
    th_al = th - al
    th_be = th - be
    ei = be - np.log(np.exp(al) + np.exp(th) + 1e-12)
    feats += [th, al, be, th_al, th_be, ei]

    return np.array(feats, dtype=float)


# =============== MI (riemann/tangent) ===============
def _vec_sym(M: np.ndarray) -> np.ndarray:
    C = M.shape[0]
    i0, i1 = np.triu_indices(C)
    v = M[i0, i1].astype(float)
    off = i0 != i1
    v[off] *= np.sqrt(2.0)
    return v

def _spd_log(C: np.ndarray) -> np.ndarray:
    from numpy.linalg import eigh
    lam, V = eigh(C)
    lam = np.clip(lam, 1e-10, None)
    return V @ np.diag(np.log(lam)) @ V.T

def _cov_spd(epoch: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    C = np.cov(epoch.T)
    C = C + ridge * np.eye(C.shape[0])
    from numpy.linalg import eigh
    lam, V = eigh(C)
    lam = np.clip(lam, 1e-10, None)
    return V @ np.diag(lam) @ V.T

def predict_mi_proba(epoch_2d: np.ndarray, model: Dict) -> float:
    fs = float(model["fs"])
    ridge = float(model["ridge_cov"])
    ep = apply_car(bandpass(apply_notch(demean(epoch_2d), fs, 50.0, 35.0), fs, 0.5, 40.0, 4))
    feats = []
    for (fmin, fmax), L_ref in zip(model["bands"], model["L_refs"]):
        ep_b = bandpass(ep, fs, fmin, fmax, 4)
        C = _cov_spd(ep_b, ridge=ridge)
        v = _vec_sym(_spd_log(C) - L_ref)
        feats.append(v)
    x = np.hstack(feats).reshape(1, -1)
    x_sc = model["scaler"].transform(x)
    return float(model["clf"].predict_proba(x_sc)[0, 1])

def predict_att_proba(epoch_2d: np.ndarray, fs: float, ch_names: list, scaler, clf) -> float:
    ep = apply_car(bandpass(apply_notch(demean(epoch_2d), fs, 50.0, 35.0), fs, 0.5, 40.0, 4))
    feats = extract_attention_features(ep, fs, ch_names).reshape(1, -1)
    x_sc = scaler.transform(feats)
    return float(clf.predict_proba(x_sc)[0, 1])


# =============== Rolling indices & gating ===============
class RollingIndex:
    def __init__(self, win_s: float):
        self.win_s = float(win_s)
        self.buf = deque()  # (t, value)

    def push(self, t: float, value: float):
        self.buf.append((t, float(value)))
        self._trim(t)

    def _trim(self, t_now: float):
        cutoff = t_now - self.win_s
        while self.buf and self.buf[0][0] < cutoff:
            self.buf.popleft()

    def mean(self) -> Optional[float]:
        if not self.buf:
            return None
        return float(np.mean([v for _, v in self.buf]))

class DecisionGate:
    def __init__(self, policy):
        self.p = policy
        self.last_cmd_time = -1e9
        self.start_streak = 0

    def reset(self):
        self.start_streak = 0
        self.last_cmd_time = -1e9

    def decide(self, t_now: float, mi_idx: Optional[float], att_idx: Optional[float]) -> Optional[int]:
        if (t_now - self.last_cmd_time) < self.p.cooldown_s:
            return None
        if mi_idx is None:
            return None
        if mi_idx <= self.p.mi_stop:
            self.last_cmd_time = t_now
            self.start_streak = 0
            return 0
        att_ok = (att_idx is not None) and (att_idx >= self.p.att_start_lo)
        hi_ok = (mi_idx >= self.p.mi_start_hi) and (not self.p.require_att_on_high or att_ok)
        lo_ok = (mi_idx >= self.p.mi_start_lo) and att_ok
        if hi_ok or lo_ok:
            self.start_streak += 1
            if self.start_streak >= self.p.consecutive_start_needed:
                self.last_cmd_time = t_now
                self.start_streak = 0
                return 1
        else:
            self.start_streak = 0
        return None


# =============== Unity & MATLAB servers ===============
class UnityServer:
    def __init__(self, ip: str, port: int, protocol):
        self.ip = ip
        self.port = port
        self.proto_len = protocol.proto_header_len_bytes
        self.proto_endian = protocol.proto_header_byteorder
        self.hdr_defaults = protocol.json_header_defaults
        self.events = protocol.events
        self.sock = None
        self.conn = None

        self.run_finished_event = threading.Event()

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.bind((self.ip, self.port))
        self.sock.listen(1)
        print(f"[Unity] Listening on {self.ip}:{self.port}")
        self.conn, addr = self.sock.accept()
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # ← también en la conexión
        print(f"[Unity] Connected: {addr}")

        threading.Thread(target=self._recv_loop, daemon=True).start()

    def _send(self, payload: dict):
        if not self.conn:
            print("[Unity] ERROR: no client connected; cannot send.")
            return
        js = json.dumps(payload).encode('utf-8')
        hdr = dict(self.hdr_defaults)
        hdr["content-length"] = str(len(js))
        hdr_bytes = json.dumps(hdr).encode('utf-8')
        n = len(hdr_bytes)
        pfx = n.to_bytes(self.proto_len, byteorder=self.proto_endian)
        pkt = pfx + hdr_bytes + js
        try:
            self.conn.sendall(pkt)
            print(f"[Unity] >> sent {len(pkt)} bytes (prefix={len(pfx)}, header={len(hdr_bytes)}, body={len(js)})")
        except Exception as e:
            print(f"[Unity] send ERROR: {e}")

    def send_set_parameters(self, settings_dict: dict, with_payload: bool = False):
        msg = {"event_type": self.events["set_parameters"]}
        if with_payload:
            msg["payload"] = settings_dict
        self._send(msg)

    def send_classification(self, value: int):
        msg = {"event_type": self.events["classification_result"], "value": int(value)}
        self._send(msg)
        # alternativo (si tu script escucha otro nombre/clave)
        alt = {"event_type": "classification", "label": int(value)}
        self._send(alt)

    def send_stop(self):
        self._send({"event_type": self.events["stop"]})

    def close(self):
        try:
            if self.conn:
                self.conn.close()
        finally:
            if self.sock:
                self.sock.close()

    def send_play(self):
        self._send({"event_type": self.events.get("play", "play")})

    def send_pause(self):
        self._send({"event_type": self.events.get("pause", "pause")})

    def send_resume(self):
        self._send({"event_type": self.events.get("resume", "resume")})

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Unity closed connection")
            buf += chunk
        return buf

    def _recv_loop(self):
        try:
            while True:
                # 1) prefijo con tamaño del header JSON
                pfx = self._recv_exact(self.proto_len)
                n_hdr = int.from_bytes(pfx, byteorder=self.proto_endian, signed=False)

                # 2) header JSON
                hdr_bytes = self._recv_exact(n_hdr)
                hdr = json.loads(hdr_bytes.decode("utf-8"))
                n_body = int(hdr.get("content-length", "0"))

                # 3) body JSON
                body = self._recv_exact(n_body)
                msg = json.loads(body.decode("utf-8"))

                evt = msg.get("event_type", "")
                # Marca fin de sesión cuando Unity lo diga
                if evt == "run_finished":
                    print("[Unity] << run_finished")
                    self.run_finished_event.set()
                else:
                    # logs de depuración opcionales
                    print(f"[Unity] << {evt} | {msg}")
        except Exception as e:
            print(f"[Unity] recv ERROR: {e}")

import socket
import threading

class MatlabServer:
    def __init__(self, host="127.0.0.1", port=9999):
        self.host = host
        self.port = int(port)
        self._srv = None
        self._conn = None
        self._lock = threading.Lock()

    def start(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, self.port))
        self._srv.listen(1)
        print(f"[MATLAB] listening on {self.host}:{self.port} ...")
        self._conn, addr = self._srv.accept()   # <- BLOQUEA hasta que MATLAB conecte
        print(f"[MATLAB] connected from {addr}")

    def is_connected(self):
        return self._conn is not None

    def send_binary(self, value: int):
        if not self._conn:
            return
        payload = f"{int(bool(value))}\n".encode("ascii")
        with self._lock:
            try:
                self._conn.sendall(payload)
            except Exception:
                try: self._conn.close()
                except Exception: pass
                self._conn = None

    def stop(self):
        try:
            if self._conn: self._conn.close()
        except Exception:
            pass
        try:
            if self._srv: self._srv.close()
        except Exception:
            pass
        self._conn = None
        self._srv = None


# =============== LSL reader ===============
class LSLReader:
    def __init__(self, target_ch_names: list, epoch_len_s: float,
                 prefer_name: Optional[str] = None, stale_after_s: float = 2.0):
        self.target_ch_names = target_ch_names
        self.epoch_len_s = float(epoch_len_s)
        self.prefer_name = prefer_name           # e.g., "Signal_generator"
        self.stale_after_s = float(stale_after_s)
        self.inlet: Optional[StreamInlet] = None
        self.fs: Optional[float] = None
        self.ch_count: Optional[int] = None
        self.buf_sec = 12.0
        self.ring = None
        self.times = None
        self.write_idx = 0
        self._last_sample_t = time.perf_counter()
        self._lock = threading.RLock()

    def get_lsl_labels(self) -> list:
        """Devuelve la lista de etiquetas de canal del stream LSL (['F3','C3',...])."""

        inlet = self.inlet
        if inlet is None:
            return []
        ch = inlet.info().desc().child('channels').child('channel')
        labels = []
        while not ch.empty():
            labels.append(ch.child_value('label'))
            ch = ch.next_sibling()
        return labels

    def _resolve_and_open(self):
        # 1) intenta por nombre
        streams = []
        if self.prefer_name:
            try:
                streams = resolve_byprop('name', self.prefer_name, timeout=1.0)
            except Exception:
                streams = []
        # 2) sino, por tipo EEG
        if not streams:
            streams = resolve_byprop('type', 'EEG', timeout=5.0)
        if not streams:
            raise RuntimeError("No LSL EEG stream found.")

        self.inlet = StreamInlet(streams[0], max_buflen=60, recover=True)
        info = self.inlet.info()
        self.fs = info.nominal_srate()
        self.ch_count = info.channel_count()
        self.name = info.name()
        print(f"[LSL] Connected. fs={self.fs:.2f} Hz, channels={self.ch_count}")

        N = int(self.buf_sec * self.fs)
        self.ring = np.zeros((N, self.ch_count), dtype=np.float64)
        self.times = np.zeros(N, dtype=np.float64)
        with self._lock:
            self.write_idx = 0
        self._last_sample_t = time.perf_counter()

    def connect(self):
        print("[LSL] Resolving EEG stream...")
        self._resolve_and_open()

    def _try_reconnect(self):
        try:
            print("[LSL] Stale stream, trying to reconnect...")
            self._resolve_and_open()
            print("[LSL] Reconnected.")
        except Exception as e:
            print(f"[LSL] Reconnect failed: {e}")

    def pull_forever(self, stop_event: threading.Event):
        while not stop_event.is_set():
            try:
                samples, timestamps = self.inlet.pull_chunk(timeout=0.25)
            except Exception:
                samples, timestamps = [], []

            if timestamps:
                self._last_sample_t = time.perf_counter()
                samples = np.asarray(samples, dtype=np.float64)
                ts = np.asarray(timestamps, dtype=np.float64)
                n = samples.shape[0]
                N = self.ring.shape[0]
                i0 = self.write_idx % N

                with self._lock:
                    if i0 + n <= N:
                        self.ring[i0:i0 + n, :] = samples
                        self.times[i0:i0 + n] = ts
                    else:
                        split = N - i0
                        self.ring[i0:N, :] = samples[:split]
                        self.times[i0:N] = ts[:split]
                        self.ring[0:n - split, :] = samples[split:]
                        self.times[0:n - split] = ts[split:]
                    self.write_idx = (self.write_idx + n) % N

            else:
                # si no hay muestras durante un rato, intenta reconectar
                if (time.perf_counter() - self._last_sample_t) > self.stale_after_s:
                    self._try_reconnect()
                time.sleep(0.05)

    def get_epoch(self) -> Optional[np.ndarray]:

        fs = self.fs
        if fs is None or self.ring is None:
            return None
        L = int(round(self.epoch_len_s * fs))
        with self._lock:
            N = self.ring.shape[0]
            end = self.write_idx % N
            start = (end - L) % N
            have = (end - start) % N
            if have < L:
                return None
            if start < end:
                return self.ring[start:end, :].copy()
            else:
                return np.vstack((self.ring[start:N, :], self.ring[0:end, :])).copy()



# =============== Clase App (interfaz MEDUSA) ===============
class App(resources.AppSkeleton):
    """
    App MEDUSA minimal:
    - Lanza Unity (opcional) y espera conexión
    - (Opcional) abre socket MATLAB
    - Lee LSL, infiere MI/ATT (5 s), aplica política y envía 0/1 (una vez por trial)
    """

    def __init__(self, app_info, app_settings, medusa_interface,
                 app_state, run_state, working_lsl_streams_info, rec_info):

        # 1) Logger desde el principio
        self.log = make_logger("App")
        self.logger = self.log  # alias por compatibilidad

        # 2) Atributos que usarás después
        self.is_debugging = False
        self.mi_model = None
        self._lsl_shims = {}
        self.eeg_worker_name = None
        self._working_lsl_streams_info = working_lsl_streams_info

        self.TAG = "[apps/exo/main]"
        self.S = None
        self._unity = None
        self._unity_proc = None
        self._matlab = None
        self._lsl = None             # <<— SIEMPRE usa _lsl (no _lsl_reader)
        self._stop_pull = None
        self._pull_thread = None
        self.stop = False

        # 3) Super
        super().__init__(app_info, app_settings, medusa_interface,
                         app_state, run_state, working_lsl_streams_info, rec_info)

        # 4) Si MEDUSA aporta logger, úsalo
        if hasattr(self, "medusa_interface"):
            for attr in ("logger", "log"):
                cand = getattr(self.medusa_interface, attr, None)
                if cand is not None and hasattr(cand, "info"):
                    self.log = cand
                    self.logger = self.log
                    break

        # 5) UI / tema (si no está disponible, no rompas)
        try:
            theme_colors = gui_utils.get_theme_colors('dark')
            self.log_color = theme_colors.get('THEME_TEXT_ACCENT', None)
        except Exception:
            self.log_color = None

        # 6) Hints para AppSkeleton
        try:
            self.app_info['has_manager_thread'] = False
        except Exception:
            pass

        # 7) Info inicial EEG
        self.eeg_worker_name = self.get_eeg_worker_name(working_lsl_streams_info)
        self.log.info("App inicializada. EEG worker (hint): %s", self.eeg_worker_name)

    # ----------------- helpers MEDUSA-like -----------------

    def handle_exception(self, ex: Exception):
        try:
            if hasattr(self, "medusa_interface") and hasattr(self.medusa_interface, "error"):
                self.medusa_interface.error(ex)
            else:
                self.log.error("Unhandled exception: %s", ex, exc_info=True)
        finally:
            try:
                self._shutdown()
            except Exception:
                pass

    def get_eeg_worker_name(self, working_lsl_streams_info) -> str | None:
        for info in (working_lsl_streams_info or []):
            if info.get('lsl_type') == 'EEG':
                return info.get('lsl_name')
        return None

    def get_lsl_worker(self):
        # Lazy-init si hace falta
        if (not self._lsl_shims) or (self._lsl is None):
            self._setup_lsl(self._working_lsl_streams_info)

        if self.eeg_worker_name in self._lsl_shims:
            return self._lsl_shims[self.eeg_worker_name]
        if self._lsl_shims:
            return next(iter(self._lsl_shims.values()))
        raise RuntimeError("No LSL shim initialized.")

    def get_eeg_data(self):
        lsl_worker = self.get_lsl_worker()
        channels = meeg.EEGChannelSet()
        channels.set_standard_montage(lsl_worker.receiver.l_cha)
        times_, signal_ = lsl_worker.get_data()
        if times_.shape[0] != signal_.shape[0]:
            m = min(times_.shape[0], signal_.shape[0])
            print(f'[get_eeg_data] Warning! trimmed to {m} samples.')
            times_ = times_[:m]
            signal_ = signal_[:m, :]
        return times_, signal_, lsl_worker.receiver.fs, channels, lsl_worker.receiver.name

    def check_settings_config(self, app_settings) -> bool:
        # Validación ligera (no bloquea)
        try:
            S = Settings()
            if S.connection.launch_unity:
                exe = S.connection.resolve_unity()
                if not os.path.isfile(exe):
                    print(f"[WARN] Unity EXE not found (launch_unity=True): {exe}")
            if S.run.mode == "online" and not S.run.mi_model_path:
                print("[WARN] ONLINE mode sin mi_model_path")
        except Exception as e:
            print("[check_settings_config] Warning:", e)
        return True

    def check_lsl_config(self, working_lsl_streams_info):
        eeg_infos = [i for i in (working_lsl_streams_info or []) if i.get('lsl_type') == 'EEG']
        if len(eeg_infos) != 1:
            self.log.error("Expected exactly 1 EEG stream, found %d", len(eeg_infos))
            return False
        self.log.info("EEG stream detected: %s", eeg_infos[0].get('lsl_name'))
        return True

    def send_to_log(self, msg: str):
        try:
            if self.log_color:
                self.medusa_interface.log(msg, {'color': self.log_color, 'font-style': 'italic'})
            else:
                self.medusa_interface.log(msg)
        except Exception:
            print(msg)

    def _stop_requested(self) -> bool:
        st = self.run_state.value
        STOP = getattr(mds_constants, "RUN_STATE_STOP", None)
        STOPPED = getattr(mds_constants, "RUN_STATE_STOPPED", None)
        return (STOP is not None and st == STOP) or (STOPPED is not None and st == STOPPED)

    # ----------------- LSL setup (único sitio donde tocas _lsl) -----------------

    def _setup_lsl(self, working_lsl_streams_info=None):
        if (self._lsl is not None) and self._lsl_shims:
            return

        prefer_name = None
        infos = working_lsl_streams_info or []
        eeg_infos = [i for i in infos if i.get('lsl_type') == 'EEG']
        if eeg_infos:
            prefer_name = eeg_infos[0].get('lsl_name')

        # Crea tu LSLReader (firma real)
        run = getattr(self, "S", None).run if getattr(self, "S", None) else None
        target_ch = (run.target_channels if run else [])
        epoch_len = (run.epoch_len_s if run else 5.0)

        self._lsl = LSLReader(
            target_ch_names=target_ch,
            epoch_len_s=epoch_len,
            prefer_name=prefer_name or "Signal_generator",
            stale_after_s=3.0,
        )

        # Conecta y arranca hilo de captura
        self._lsl.connect()
        if self._stop_pull is None:
            self._stop_pull = threading.Event()
        if (self._pull_thread is None) or (not self._pull_thread.is_alive()):
            self._pull_thread = threading.Thread(
                target=self._lsl.pull_forever, args=(self._stop_pull,), daemon=True
            )
            self._pull_thread.start()

        # Registra shim en TU dict privado
        eeg_name = getattr(self._lsl, "name", None) or prefer_name or "Signal_generator"
        self._lsl_shims = {eeg_name: LSLWorkerShim(self._lsl, stream_name=eeg_name)}
        self.eeg_worker_name = eeg_name
        self.log.info("LSL inicializado con stream '%s'", self.eeg_worker_name)

    # ----------------- Entry point -----------------

    def main(self):
        self.medusa_interface.app_state_changed(mds_constants.APP_STATE_POWERING_ON)
        self.send_to_log("[BOOT] main() starting…")
        self.send_to_log("[BOOT] Settings about to validate…")

        try:
            # --- Settings ---
            self.S = Settings()
            self.S.validate()

            # --- Launch Unity (optional) ---
            if self.S.connection.launch_unity:
                import subprocess
                exe = self.S.connection.resolve_unity()
                if not os.path.isfile(exe):
                    raise FileNotFoundError(f"No encuentro Unity EXE: {exe}")
                self._unity_proc = subprocess.Popen(
                    [exe, self.S.connection.ip, str(self.S.connection.unity_port)],
                    cwd=os.path.dirname(exe)
                )

            # --- Unity server ---
            self._unity = UnityServer(self.S.connection.ip, self.S.connection.unity_port, self.S.protocol)
            unity_ready = threading.Event()
            threading.Thread(target=lambda: (self._unity.start(), unity_ready.set()), daemon=True).start()

            # --- MATLAB (optional) ---
            matlab_ready = threading.Event()
            if self.S.thresholds.send_binary_to_matlab:
                self._matlab = MatlabServer("127.0.0.1", 9999)
                threading.Thread(target=lambda: (self._matlab.start(), matlab_ready.set()), daemon=True).start()
                self.send_to_log("[BOOT] MATLAB enabled, spawning server thread @ 127.0.0.1:9999")

            # --- LSL setup (always use self._lsl) ---
            self._setup_lsl(self._working_lsl_streams_info)

            # --- Labels from LSL (Signal Generator usually has none) ---
            lsl_labels = None
            try:
                lsl_labels = self._lsl.get_lsl_labels()
            except Exception:
                lsl_labels = None

            target = list(self.S.run.target_channels)
            if not lsl_labels:
                n = getattr(self._lsl, "ch_count", None)
                if n is None and getattr(self._lsl, "inlet", None):
                    try:
                        n = self._lsl.inlet.info().channel_count()
                    except Exception:
                        n = None
                n = n or len(target) or 8
                lsl_labels = [f"Ch{i + 1}" for i in range(n)]
                self.log.warning("LSL sin labels → usando sintéticas: %s", lsl_labels)

            missing = [ch for ch in target if ch not in lsl_labels]
            if missing:
                self.log.warning("Canales objetivo no están en el stream: %s (stream=%s)", missing, lsl_labels)
                reorder_idx = None
            else:
                reorder_idx = [lsl_labels.index(ch) for ch in target]

            fs = float(getattr(self._lsl, "fs", 256.0) or 256.0)

            # --- Pull thread ---
            if self._stop_pull is None:
                self._stop_pull = threading.Event()
            if (self._pull_thread is None) or (not self._pull_thread.is_alive()):
                self._pull_thread = threading.Thread(target=self._lsl.pull_forever, args=(self._stop_pull,),
                                                     daemon=True)
                self._pull_thread.start()

            # --- Wait sockets ---
            while not unity_ready.is_set():
                if self._stop_requested():
                    try:
                        if self._unity: self._unity.send_stop()
                    finally:
                        return
                time.sleep(0.05)
            if self.S.thresholds.send_binary_to_matlab:
                print("[WAIT] Waiting for MATLAB client (accept)…")
                while not matlab_ready.is_set():
                    if self._stop_requested():
                        try:
                            if self._unity: self._unity.send_stop()
                        finally:
                            return
                    time.sleep(0.05)

            # --- READY → PLAY ---
            self.medusa_interface.app_state_changed(mds_constants.APP_STATE_ON)
            self.run_state.value = mds_constants.RUN_STATE_READY
            while self.run_state.value != mds_constants.RUN_STATE_RUNNING:
                if self._stop_requested():
                    try:
                        if self._unity: self._unity.send_stop()
                    finally:
                        return
                time.sleep(0.05)

            # --- Parameters & PLAY ---
            self._unity.send_set_parameters(self.S.to_dict(), with_payload=False)
            self._unity.send_play()

            # --- Models ---
            mi_model = load(self.S.run.resolve(self.S.run.mi_model_path))
            att_scaler = att_clf = None
            if self.S.run.use_att_model:
                att_clf = load(self.S.run.resolve(self.S.run.att_model_path))
                att_scaler = load(self.S.run.resolve(self.S.run.att_scaler_path))
                if not hasattr(att_clf, "predict_proba"):
                    raise TypeError("El clasificador ATT debe exponer .predict_proba().")

            # --- Policy (batch mode) ---
            n_trials = self.S.timings.n_trials
            dt_eval = 1.0 / max(1.0, self.S.thresholds.decision_rate_hz)
            # gate no es necesario en batch; lo dejamos fuera

            print(self.TAG, f"Run START. Trials={n_trials}")
            for trial in range(1, n_trials + 1):
                if self._stop_requested():
                    try:
                        if self._unity: self._unity.send_stop()
                    finally:
                        break

                t0 = time.perf_counter()

                # PREP
                while (time.perf_counter() - t0) < (self.S.timings.o_trial_cue / 1000.0):
                    if self._stop_requested():
                        if self._unity: self._unity.send_stop()
                        break
                    time.sleep(0.01)
                if self._stop_requested():
                    break

                # CUE
                while (time.perf_counter() - t0) < (self.S.timings.o_trial_start_feedback / 1000.0):
                    if self._stop_requested():
                        if self._unity: self._unity.send_stop()
                        break
                    time.sleep(0.01)
                if self._stop_requested():
                    break

                # FEEDBACK (batch per trial)
                sent_this_trial = False
                t_fb_start = t0 + (self.S.timings.o_trial_start_feedback / 1000.0)
                t_fb_end = t0 + (self.S.timings.o_trial_final_feedback / 1000.0)

                # Reset Unity to ensure 0→1 edge
                if self.S.thresholds.send_binary_to_unity:
                    self._unity.send_classification(0)
                    # self.send_to_log("[SYNC] trial reset → 0 enviado a Unity")

                if self.S.thresholds.send_binary_to_matlab and self._matlab:
                    self._matlab.send_binary(0)

                # Grace time so Unity & scene are ready
                t_eval_start = t_fb_start + 0.5  # 0.3–0.5 s is usually fine

                # Accumulate model outputs over the whole feedback window
                mi_vals, att_vals = [], []

                next_eval = time.perf_counter()
                while time.perf_counter() < t_fb_end and not self._stop_requested():
                    if time.perf_counter() >= next_eval:
                        next_eval += dt_eval

                        # Wait until grace window has passed
                        now = time.perf_counter()
                        if now < t_eval_start:
                            if next_eval < t_eval_start:
                                next_eval = t_eval_start
                            time.sleep(0.001)
                            continue

                        epoch = self._lsl.get_epoch()
                        #if epoch is None:
                        #    if int(time.perf_counter() * 2) % 4 == 0:
                        #        self.send_to_log("[LSLReader] aún sin epoch suficiente…")
                        #    continue
                        #else:
                        #    if int(time.perf_counter() * 2) % 4 == 0:
                        #        self.send_to_log("[LSLReader] epoch OK")

                        if reorder_idx is not None:
                            epoch = epoch[:, reorder_idx]

                        p_mi = predict_mi_proba(epoch, mi_model)
                        p_att = None
                        if self.S.run.use_att_model and att_scaler is not None and att_clf is not None:
                            p_att = predict_att_proba(epoch, fs, target, att_scaler, att_clf)

                        mi_vals.append(p_mi)
                        if p_att is not None:
                            att_vals.append(p_att)

                        # debug log
                        #self.send_to_log(f"[DEC] p_mi={p_mi:.3f}" + ("" if p_att is None else f" | p_att={p_att:.3f}"))

                    time.sleep(0.001)

                # ---- End of FEEDBACK → single decision (batch) ----
                mi_mean = float(np.mean(mi_vals)) if mi_vals else None
                att_mean = float(np.mean(att_vals)) if att_vals else None

                decision = None
                if mi_mean is not None:
                    if mi_mean <= self.S.thresholds.mi_stop:
                        decision = 0
                    else:
                        att_ok = (att_mean is not None) and (att_mean >= self.S.thresholds.att_start_lo)
                        hi_ok = (mi_mean >= self.S.thresholds.mi_start_hi) and (
                                    not self.S.thresholds.require_att_on_high or att_ok)
                        lo_ok = (mi_mean >= self.S.thresholds.mi_start_lo) and att_ok
                        if hi_ok or lo_ok:
                            decision = 1

                self.send_to_log(
                    f"Trial {trial} - p_mi = {('None' if mi_mean is None else f'{mi_mean:.3f}')}"
                    + ("" if att_mean is None else f" | p_att = {att_mean:.3f}")
                )

                if decision is not None:
                    if self.S.thresholds.send_binary_to_unity:
                        self._unity.send_classification(int(decision))
                    if self.S.thresholds.send_binary_to_matlab and self._matlab:
                        self._matlab.send_binary(int(decision))
                    att_txt = "None" if att_mean is None else f"{att_mean:.3f}"
                    self.send_to_log(
                        f"Trial {trial} - Decision = {int(decision)}")
                    sent_this_trial = True
                else:
                    if self.S.thresholds.send_binary_to_unity:
                        self._unity.send_classification(0)
                    if self.S.thresholds.send_binary_to_matlab and self._matlab:
                        self._matlab.send_binary(0)
                    self.send_to_log(f"Trial {trial} - Decision = 0")

                self.send_to_log("=========================================")

                if self._stop_requested():
                    if self._unity: self._unity.send_stop()
                    break

                # REST
                rest_until = time.perf_counter() + (self.S.timings.t_trial_rest / 1000.0)
                while time.perf_counter() < rest_until:
                    if self._stop_requested():
                        if self._unity: self._unity.send_stop()
                        break
                    time.sleep(0.01)
                if self._stop_requested():
                    if self._unity: self._unity.send_stop()
                    break

            # Normal end
            if self._unity:
                waited = self._unity.run_finished_event.wait(timeout=15.0)  # 5s de score + margen
                if not waited:
                    self.send_to_log("Run finished")
                self._unity.send_stop()

        except Exception as ex:
            try:
                self.medusa_interface.error(ex)
            except Exception:
                pass

        finally:
            self._shutdown()

    # ----------------- Manager thread (no-op) -----------------
    def manager_thread_worker(self, *args, **kwargs):
        try:
            while True:
                if getattr(self.app_state, "value", None) == mds_constants.APP_STATE_OFF:
                    break
                st = getattr(self.run_state, "value", None)
                STOP = getattr(mds_constants, "RUN_STATE_STOP", None)
                STOPPED = getattr(mds_constants, "RUN_STATE_STOPPED", None)
                if st in (STOP, STOPPED):
                    break
                time.sleep(0.2)
        except Exception as e:
            try:
                self.medusa_interface.error(e)
            except Exception:
                pass

    def process_event(self, msg: dict): pass
    def nf_calibration(self): pass

    def _shutdown(self):
        self.medusa_interface.app_state_changed(mds_constants.APP_STATE_POWERING_OFF)
        try:
            if self._stop_pull: self._stop_pull.set()
            if self._pull_thread: self._pull_thread.join(timeout=1.0)
        except Exception: pass

        try:
            if getattr(self, "_matlab", None):
                self._matlab.stop()
        except Exception:
            pass

        try:
            if self._matlab: self._matlab.close()
        except Exception: pass
        try:
            if self._unity: self._unity.close()
        except Exception: pass
        try:
            if self._unity_proc and (self._unity_proc.poll() is None):
                for _ in range(20):
                    if self._unity_proc.poll() is not None: break
                    time.sleep(0.1)
                if self._unity_proc.poll() is None:
                    self._unity_proc.terminate()
                    for _ in range(20):
                        if self._unity_proc.poll() is not None: break
                        time.sleep(0.1)
                if self._unity_proc.poll() is None:
                    self._unity_proc.kill()
        except Exception: pass
        try:
            self.run_state.value = getattr(mds_constants, "RUN_STATE_STOP", self.run_state.value)
        except Exception: pass
        self.medusa_interface.app_state_changed(mds_constants.APP_STATE_OFF)

# Standalone smoke-test (no MEDUSA)
if __name__ == "__main__":
    S = Settings(); S.validate()
    print("[Standalone] Este módulo está diseñado para ser lanzado por MEDUSA (AppSkeleton).")
