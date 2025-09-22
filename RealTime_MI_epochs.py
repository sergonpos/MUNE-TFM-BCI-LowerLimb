
# docs = load_all_bson(r"C:\Users\SGP02\MEDUSA\v2024\data")

from pathlib import Path
from bson import loads
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch, detrend
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from joblib import dump
import warnings
warnings.filterwarnings("ignore")

# --- Configuración de tus tramos y etiquetas ---
SEGMENTS_S = [(5,125), (130,250), (255,375), (380,500)]
LABELS     = ['MI',     'BL',      'MI',      'BL'     ]
EPOCH_LEN_S = 5  # num segundos

def load_all_bson(folder, pattern="*.bson"):
    """Lee todos los BSON del folder y devuelve lista de (doc, filename)."""
    folder = Path(folder)
    docs = []
    for fp in sorted(folder.glob(pattern)):
        try:
            with open(fp, "rb") as f:
                doc = loads(f.read())
            docs.append((doc, fp.name))
        except Exception as e:
            print(f"[WARN] No pude leer {fp.name}: {e}")
    print(f"Leídos {len(docs)} archivos BSON.")
    return docs

def ensure_time_first(arr):
    """Asegura forma (n_muestras, n_canales)."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("La señal EEG debe ser 2D.")
    # Si viene como (n_canales, n_muestras), transpón
    if arr.shape[0] < arr.shape[1]:
        return arr.T
    return arr

def epochar_segment(seg_data, fs, epoch_len_s=5):
    """
    Corta un segmento continuo en épocas no solapadas.
    Devuelve epochs (n_ep, L, C) y los tiempos de inicio de cada época (s).
    """
    seg_data = ensure_time_first(seg_data)
    L = int(round(epoch_len_s * fs))
    n = seg_data.shape[0] // L
    if n == 0:
        return np.empty((0, L, seg_data.shape[1])), np.empty((0,))
    seg_data = seg_data[:n*L, :]
    epochs = seg_data.reshape(n, L, seg_data.shape[1])
    epoch_starts_s = np.arange(n) * epoch_len_s
    return epochs, epoch_starts_s

def process_doc(doc, source_name, segments_s=SEGMENTS_S, labels=LABELS, epoch_len_s=EPOCH_LEN_S):
    """
    Para un archivo BSON:
      - extrae EEG, corta en 4 tramos, epoca 5s,
      - devuelve X (numpy), y (labels) y meta (DataFrame).
    """
    eeg = doc["eeg"]
    fs = float(eeg["fs"])
    signal = ensure_time_first(eeg["signal"])  # (muestras, canales)
    n_channels = signal.shape[1]

    X_list, y_list, meta_rows = [], [], []

    for seg_idx, ((a, b), lab) in enumerate(zip(segments_s, labels), start=1):
        i0, i1 = int(round(a * fs)), int(round(b * fs))
        seg = signal[i0:i1, :]
        ep, ep_starts = epochar_segment(seg, fs, epoch_len_s)
        if ep.size == 0:
            continue

        # Metadatos por época
        for k in range(ep.shape[0]):
            meta_rows.append({
                "file": source_name,
                "trial_idx": seg_idx,           # 1..4
                "label": lab,                   # 'MI' / 'BL'
                "epoch_idx": k,                 # 0..(n_epocas-1)
                "epoch_start_s": float(a + ep_starts[k]),
                "epoch_end_s": float(a + ep_starts[k] + epoch_len_s),
                "fs": fs,
                "n_channels": n_channels
            })

        X_list.append(ep)
        y_list.extend([lab] * ep.shape[0])

    if not X_list:
        return np.empty((0, int(round(epoch_len_s*fs)), 0)), np.empty((0,)), pd.DataFrame()

    X = np.concatenate(X_list, axis=0)   # (N_ep, L, C)
    y = np.array(y_list, dtype=object)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta

def process_folder(folder, pattern="*.rec.bson", epoch_len_s=EPOCH_LEN_S):
    """Procesa todos los BSON: concatena X, y y meta de todos los archivos."""
    docs = load_all_bson(folder, pattern)
    X_all, y_all, metas = [], [], []
    for doc, name in docs:
        X, y, meta = process_doc(doc, name, epoch_len_s=epoch_len_s)
        if X.size:
            X_all.append(X)
            y_all.append(y)
            metas.append(meta)

    if not X_all:
        print("No se generaron épocas.")
        return np.empty((0,)), np.empty((0,)), pd.DataFrame()

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    meta_all = pd.concat(metas, ignore_index=True)
    return X_all, y_all, meta_all

# --------- USO ----------
# Cambia la ruta a tu carpeta
folder = r"C:\Users\SGP02\MEDUSA\v2024\data"

X, y, meta = process_folder(folder, pattern="*.rec.bson", epoch_len_s=5)

# =================== CLASIFICAR N ÉPOCAS ALEATORIAS ===================
import numpy as np
from joblib import load
from numpy.linalg import eigh
from scipy.signal import iirnotch, butter, filtfilt, welch
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import pandas as pd

# --- Configuración ---
TASK = 'MI'      # 'MI' o 'ATT' (este flag ya no afecta a MI: siempre usaremos el modelo riemanniano)
N_SAMPLES = 50   # nº de épocas aleatorias
RANDOM_STATE = 41

# Rutas de modelos
RIEM_MODEL_PATH_MI = "model_riem_tangent_lda_full.joblib"  # << nuevo modelo MI (Tangent+LDA)
MODEL_PATH_ATT = "model_att_frontal_lda_full.joblib"       # si lo tienes
SCALER_PATH_ATT = "scaler_att_frontal_full.joblib"         # si lo tienes

# --- Cargar modelo MI (riemanniano Tangent+LDA) ---
riem_model_mi = load(RIEM_MODEL_PATH_MI)
MI_DEFAULT_THR = float(riem_model_mi.get("default_threshold", 0.5))  # por si no estaba guardado, usa 0.5
print(f"[INFO] Modelo MI (Riemann Tangent+LDA): {RIEM_MODEL_PATH_MI} | default_threshold={MI_DEFAULT_THR:.3f}")

# --- Cargar modelo ATT (igual que antes) ---
clf_att  = load(MODEL_PATH_ATT)
scaler_att = load(SCALER_PATH_ATT)
print(f"[INFO] Modelo ATT: {MODEL_PATH_ATT} | Scaler ATT: {SCALER_PATH_ATT}")

# --- Montaje / fs ---
FS = float(meta["fs"].unique()[0])
CH_NAMES = ['F3','C3','P3','Fz','Cz','F4','C4','P4'] if X.shape[2]==8 else [f"CH{i}" for i in range(X.shape[2])]

# --- Etiquetas binarias ---
y_mi_bin  = (y == 'MI').astype(int)   # 1=MI, 0=BL
y_att_bin = y_mi_bin.copy()           # 1=ATT, 0=REST (idéntico en tu diseño actual)

# --- Preprocesado común que ya tenías ---
def demean(x): return x - np.mean(x, axis=0, keepdims=True)
def apply_notch(x, fs, f0=50.0, Q=35.0):
    b, a = iirnotch(w0=f0/(fs/2), Q=Q); return filtfilt(b, a, x, axis=0)
def apply_bandpass(x, fs, fmin=0.5, fmax=40.0, order=4):
    b, a = butter(order, [fmin/(fs/2), fmax/(fs/2)], btype='band'); return filtfilt(b, a, x, axis=0)

# --- Utilidades ATT (sin cambios) ---
FRONTAL = ['F3','Fz','F4']
BANDS_ATT = {'theta': (4,7), 'alpha': (8,12), 'betaL': (13,20)}

def car(epoch_2d):
    return epoch_2d - np.mean(epoch_2d, axis=1, keepdims=True)

def band_logpower(sig_1d, fs, fmin, fmax, win_s=1.0, overlap=0.5):
    nperseg = int(round(win_s*fs)); noverlap = int(round(nperseg*overlap))
    f, Pxx = welch(sig_1d, fs=fs, nperseg=nperseg, noverlap=noverlap)
    m = (f>=fmin)&(f<=fmax)
    p = np.trapz(Pxx[m], f[m]) if np.any(m) else 0.0
    return np.log(p + 1e-12)

def att_features_epoch(epoch_2d, fs, ch_names):
    ep = car(epoch_2d)
    ch2idx = {ch:i for i, ch in enumerate(ch_names)}
    have = [ch for ch in FRONTAL if ch in ch2idx]
    if not have:
        have = [ch_names[i] for i in range(min(3, len(ch_names)))]
        ch2idx = {ch:i for i, ch in enumerate(ch_names)}
    feats = []
    for ch in have:
        sig = ep[:, ch2idx[ch]]
        th = band_logpower(sig, fs, *BANDS_ATT['theta'])
        al = band_logpower(sig, fs, *BANDS_ATT['alpha'])
        be = band_logpower(sig, fs, *BANDS_ATT['betaL'])
        feats += [th, al, be, th - al, th - be, be - np.log(np.exp(al) + np.exp(th) + 1e-12)]
    idxs = [ch2idx[ch] for ch in have]
    sig_avg = np.mean(ep[:, idxs], axis=1)
    th = band_logpower(sig_avg, fs, *BANDS_ATT['theta'])
    al = band_logpower(sig_avg, fs, *BANDS_ATT['alpha'])
    be = band_logpower(sig_avg, fs, *BANDS_ATT['betaL'])
    feats += [th, al, be, th - al, th - be, be - np.log(np.exp(al) + np.exp(th) + 1e-12)]
    return np.array(feats, dtype=float)

# --- Utilidades RIEMANN para MI (Tangent space) ---
def cov_spd(epoch, ridge=1e-6):
    C = np.cov(epoch.T) + ridge * np.eye(epoch.shape[1])
    # clamp por seguridad antes de log
    lam, V = eigh(C)
    lam = np.clip(lam, 1e-10, None)
    return V @ np.diag(lam) @ V.T

def spd_log(C):
    lam, V = eigh(C)
    lam = np.clip(lam, 1e-10, None)
    return V @ np.diag(np.log(lam)) @ V.T

def vec_sym(M):
    idx = np.triu_indices_from(M)
    v = M[idx].astype(float)
    off = idx[0] != idx[1]
    v[off] *= np.sqrt(2.0)  # isometría Frobenius
    return v

def predict_proba_tangent_lda(epoch_2d, model):
    """
    epoch_2d: (L, C) cruda (sin preprocesado externo).
    Devuelve probabilidad de clase 'MI' usando el paquete guardado en joblib.
    """
    fs = float(model["fs"])
    # preprocesado base
    ep = demean(epoch_2d)
    ep = apply_notch(ep, fs, 50.0, 35.0)
    ep = apply_bandpass(ep, fs, 0.5, 40.0, 4)
    ep = ep - np.mean(ep, axis=1, keepdims=True)  # CAR

    feats = []
    for (fmin, fmax), L_ref in zip(model["bands"], model["L_refs"]):
        ep_b = apply_bandpass(ep, fs, fmin, fmax, 4)
        C = cov_spd(ep_b, ridge=float(model["ridge_cov"]))
        v = vec_sym(spd_log(C) - L_ref)
        feats.append(v)
    x = np.hstack(feats).reshape(1, -1)
    x_sc = model["scaler"].transform(x)
    proba = model["clf"].predict_proba(x_sc)[0, 1]
    return float(proba)

# --- Elegir N épocas aleatorias (sin reemplazo) ---
rng = np.random.default_rng(RANDOM_STATE)
if N_SAMPLES > X.shape[0]:
    raise ValueError("N_SAMPLES mayor que el nº de épocas disponibles.")
idx_pick = rng.choice(X.shape[0], size=N_SAMPLES, replace=False)

# --- Clasificar esas épocas ---
rows = []
probas_mi, preds_mi, gts_mi = [], [], []
probas_att, preds_att, gts_att = [], [], []

for i in idx_pick:
    ep_raw = X[i]  # (L, C) sin preprocesar

    # ===== MI con RIEMANN (Tangent+LDA) =====
    p_mi = predict_proba_tangent_lda(ep_raw, riem_model_mi)
    yhat_mi = int(p_mi >= MI_DEFAULT_THR)
    gt_mi = int(y_mi_bin[i])

    # ===== ATT (tu pipeline frontal) =====
    # preprocesado común para ATT
    ep_f = demean(ep_raw)
    ep_f = apply_notch(ep_f, FS, 50.0, 35.0)
    ep_f = apply_bandpass(ep_f, FS, 0.5, 40.0, 4)

    feat_att = att_features_epoch(ep_f, FS, CH_NAMES).reshape(1, -1)
    feat_att_sc = scaler_att.transform(feat_att)
    p_att = float(clf_att.predict_proba(feat_att_sc)[0, 1])
    yhat_att = int(p_att >= 0.5)
    gt_att = int(y_att_bin[i])

    # Acumular métricas
    probas_mi.append(p_mi);   preds_mi.append(yhat_mi);   gts_mi.append(gt_mi)
    probas_att.append(p_att); preds_att.append(yhat_att); gts_att.append(gt_att)

    rows.append({
        "file": meta.loc[i, "file"],
        "t_start": meta.loc[i, "epoch_start_s"],
        "t_end": meta.loc[i, "epoch_end_s"],
        "gt_label": str(y[i]),
        "p_MI": p_mi, "pred_MI": yhat_mi, "gt_MI": gt_mi,
        "p_ATT": p_att, "pred_ATT": yhat_att, "gt_ATT": gt_att,
    })

df_out = pd.DataFrame(rows).sort_values(["file", "t_start"]).reset_index(drop=True)
print("\n=== Predicciones aleatorias (MI RIEMANN + ATT) ===")
print(df_out)

def safe_auc(ytrue, p):
    try: return roc_auc_score(ytrue, p)
    except: return np.nan

# --- MI ---
acc_mi = accuracy_score(gts_mi, preds_mi)
auc_mi = safe_auc(gts_mi, probas_mi)
cm_mi = confusion_matrix(gts_mi, preds_mi)
rep_mi = classification_report(gts_mi, preds_mi, target_names=['BL','MI'])
print("\n=== Métricas (MI - Riemann Tangent+LDA) ===")
print(f"Accuracy: {acc_mi:.3f} | ROC-AUC: {auc_mi:.3f} | thr={MI_DEFAULT_THR:.3f}")
print("Confusión (MI):\n", cm_mi)
print("Reporte (MI):\n", rep_mi)

# --- ATT ---
acc_att = accuracy_score(gts_att, preds_att)
auc_att = safe_auc(gts_att, probas_att)
cm_att = confusion_matrix(gts_att, preds_att)
rep_att = classification_report(gts_att, preds_att, target_names=['REST','ATT'])
print("\n=== Métricas (ATT) ===")
print(f"Accuracy: {acc_att:.3f} | ROC-AUC: {auc_att:.3f}")
print("Confusión (ATT):\n", cm_att)
print("Reporte (ATT):\n", rep_att)



# =================== PAPER-STYLE CONTROL POLICY (5s indices + 5s cooldown) ===================
import numpy as np
import pandas as pd

PAPER_PARAMS = {
    "index_window_s": 5.0,    # <-- AHORA 5 s (antes 10 s)
    "cooldown_s": 5.0,        # "During 5 s, new commands cannot be issued."
    # Start gait (when subject is standing):
    # If MI_idx >= 0.7 OR (MI_idx >= 0.6 AND ATT_idx >= 0.4) -> MOVE
    "mi_start_hi": 0.70,
    "mi_start_lo": 0.60,
    "att_start_lo": 0.40,
    # Stop gait (when subject is walking):
    # If MI_idx <= 0.4 -> STOP
    "mi_stop": 0.40,
    # Initial state
    "initial_state": "STANDING",
    # --- Endurecimiento opcional del arranque ---
    "require_att_on_high": True,   # si True, también exige ATT≥att_start_lo cuando MI≥mi_start_hi
    "consecutive_start_needed": 1, # nº de decisiones consecutivas cumpliendo start_ok
}

def _time_weighted_index(g_times, g_probs, window_start, window_end):
    """
    Compute time-weighted average probability over [window_start, window_end]
    given piecewise-constant probabilities defined on intervals [t_start, t_end].
    g_times: array of shape (n, 2) with columns [t_start, t_end]
    g_probs: array of shape (n,) with the probability for each interval
    """
    overlap_start = np.maximum(g_times[:, 0], window_start)
    overlap_end   = np.minimum(g_times[:, 1], window_end)
    w = np.clip(overlap_end - overlap_start, a_min=0.0, a_max=None)
    if np.sum(w) <= 0:
        return np.nan
    return float(np.sum(w * g_probs) / np.sum(w))

def _compute_indices_10s(group, window_s=5.0):
    """
    Mantengo el nombre por compatibilidad con tu script original,
    pero 'window_s' ahora viene de PAPER_PARAMS (5.0s).
    """
    g = group.copy().reset_index(drop=True)
    times = g[["t_start", "t_end"]].to_numpy(dtype=float)
    p_mi  = g["p_MI"].to_numpy(dtype=float)
    p_att = g["p_ATT"].to_numpy(dtype=float)

    mi_idx = np.empty(len(g), dtype=float)
    att_idx = np.empty(len(g), dtype=float)

    for i in range(len(g)):
        w_end = float(g.loc[i, "t_end"])
        w_start = w_end - window_s
        mask = times[:, 1] > w_start
        sel_times = times[mask]
        sel_p_mi  = p_mi[mask]
        sel_p_att = p_att[mask]

        mi_idx[i]  = _time_weighted_index(sel_times, sel_p_mi,  w_start, w_end)
        att_idx[i] = _time_weighted_index(sel_times, sel_p_att, w_start, w_end)

    g["MI_idx_10s"]  = mi_idx   # columnas mantienen el nombre para no romper tu preview
    g["ATT_idx_10s"] = att_idx
    return g

def apply_paper_control_policy(df_preds, params=PAPER_PARAMS):
    """
    df_preds: like df_out (columns: file, t_start, t_end, p_MI, p_ATT, gt_label ...)
    Returns: df_ctrl with indices, control state and issued commands per row.
    """
    df = df_preds.sort_values(["file", "t_start", "t_end"]).reset_index(drop=True).copy()

    # Compute rolling indices per file usando la ventana de params["index_window_s"] (5 s)
    blocks = []
    for fname, g in df.groupby("file", sort=False):
        g2 = _compute_indices_10s(g, params["index_window_s"])
        blocks.append(g2)
    df = pd.concat(blocks, ignore_index=True)

    # Inicialización
    df["state"]   = ""       # 'STANDING' / 'WALKING'
    df["command"] = ""       # '', 'MOVE', 'STOP'
    df["cooldown_active"] = False

    consec_start_ok = {}     # confirmación consecutiva por archivo
    last_cmd_time = {}
    current_state = {}

    # State machine por archivo con cooldown + confirmación consecutiva opcional
    for fname, g in df.groupby("file", sort=False):
        idx = g.index
        state = params["initial_state"]
        last_t = -1e9
        consec_start_ok[fname] = 0

        for i in idx:
            t_now  = float(df.at[i, "t_end"])  # decisión al final de cada época
            mi_idx = float(df.at[i, "MI_idx_10s"])
            att_idx= float(df.at[i, "ATT_idx_10s"])

            cmd = ""
            in_cooldown = (t_now - last_t) < params["cooldown_s"]

            # Reglas START (con endurecimiento opcional)
            start_hi = (mi_idx >= params["mi_start_hi"])
            if params["require_att_on_high"]:
                start_hi = start_hi and (att_idx >= params["att_start_lo"])
            start_lo = (mi_idx >= params["mi_start_lo"]) and (att_idx >= params["att_start_lo"])
            want_start = start_hi or start_lo

            if state == "STANDING":
                consec_start_ok[fname] = consec_start_ok[fname] + 1 if want_start else 0
                if (not in_cooldown) and (consec_start_ok[fname] >= params["consecutive_start_needed"]):
                    cmd = "MOVE"
                    state = "WALKING"
                    last_t = t_now
                    consec_start_ok[fname] = 0
            else:
                # STOP
                stop_ok = (mi_idx <= params["mi_stop"])
                if stop_ok and not in_cooldown:
                    cmd = "STOP"
                    state = "STANDING"
                    last_t = t_now
                    consec_start_ok[fname] = 0

            df.at[i, "state"] = state
            df.at[i, "command"] = cmd
            df.at[i, "cooldown_active"] = in_cooldown

        last_cmd_time[fname] = last_t
        current_state[fname] = state

    # ----- Resumen global + por archivo (robusto en tipos) -----
    def to_num(s): return pd.to_numeric(s, errors="coerce")

    total_time_s = (to_num(df["t_end"]) - to_num(df["t_start"])).fillna(0).sum()
    total_minutes = float(total_time_s) / 60.0 if total_time_s > 0 else 0.0

    moves = int((df["command"] == "MOVE").sum())
    stops = int((df["command"] == "STOP").sum())
    fp_moves = int(((df["command"] == "MOVE") & (df.get("gt_label","") == "BL")).sum()) if "gt_label" in df.columns else np.nan

    # Tiempo caminando
    mask_w = df["state"] == "WALKING"
    on_time_s = float((to_num(df.loc[mask_w, "t_end"]) - to_num(df.loc[mask_w, "t_start"])).fillna(0).sum())

    # Mensajes (ASCII-safe para consolas Windows)
    req_att_hi = (
        f"AND ATT>={params['att_start_lo']:.2f}"
        if params["require_att_on_high"]
        else f"(no ATT check on MI>={params['mi_start_hi']:.2f})"
    )

    print(f"\n=== PAPER CONTROL POLICY ({params['index_window_s']:.0f} s indices, {params['cooldown_s']:.0f} s cooldown) ===")
    print(
        "Start when: (MI>={:.2f} {}) OR (MI>={:.2f} & ATT>={:.2f})"
        .format(params['mi_start_hi'], req_att_hi, params['mi_start_lo'], params['att_start_lo'])
    )
    print("Consecutive confirmations: {} | Stop when: MI<={:.2f}".format(
        params['consecutive_start_needed'], params['mi_stop']
    ))
    print("Commands -> MOVE: {}, STOP: {} | FP MOVE (if gt available): {}".format(
        moves, stops, fp_moves
    ))
    if total_minutes > 0:
        print("Approx. walking time: {:.1f} s over {:.2f} min".format(on_time_s, total_minutes))


    # Resumen por archivo
    print("\nPer-file summary:")
    summ = df.assign(dur=(to_num(df["t_end"]) - to_num(df["t_start"]))).groupby("file").apply(
        lambda g: pd.Series({
            "MOVE": int((g["command"]=="MOVE").sum()),
            "STOP": int((g["command"]=="STOP").sum()),
            "walk_time_s": float((to_num(g.loc[g['state']=='WALKING','t_end']) -
                                  to_num(g.loc[g['state']=='WALKING','t_start'])).fillna(0).sum())
        })
    )
    print(summ)

    # Devolver DF con nombres consistentes
    df = df.rename(columns={"MI_idx_10s":"MI_index","ATT_idx_10s":"ATT_index"})
    return df

# ---- Ejecuta la política sobre tu df_out actual ----
df_ctrl = apply_paper_control_policy(df_out, PAPER_PARAMS)

# Vista rápida
print("\nPreview:")
print(df_ctrl[["file","t_start","t_end","p_MI","p_ATT","MI_index","ATT_index","state","command","cooldown_active"]].head(15))


# =================== GRID-SEARCH: maximize walking time with FP_MOVE/min ≤ 1.0 ===================
import itertools, contextlib, sys, os
import pandas as pd
import numpy as np

# Silenciar prints internos de apply_paper_control_policy durante el barrido
@contextlib.contextmanager
def suppress_stdout():
    saved = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        try:
            sys.stdout.close()
        finally:
            sys.stdout = saved

def run_policy_with(df_preds, base_params, **overrides):
    """Ejecuta la política con params=base_params+overrides y devuelve DF control + métricas."""
    params = dict(base_params)
    params.update(overrides)
    # Por petición del usuario: confirmaciones consecutivas = 1
    params["consecutive_start_needed"] = 1
    with suppress_stdout():
        df_ctrl_local = apply_paper_control_policy(df_preds, params)
    # Métricas
    to_num = lambda s: pd.to_numeric(s, errors="coerce")
    total_time_s = (to_num(df_ctrl_local["t_end"]) - to_num(df_ctrl_local["t_start"])).fillna(0).sum()
    total_min = float(total_time_s) / 60.0 if total_time_s > 0 else 0.0
    moves = int((df_ctrl_local["command"] == "MOVE").sum())
    stops = int((df_ctrl_local["command"] == "STOP").sum())
    fp_moves = int(((df_ctrl_local["command"] == "MOVE") & (df_ctrl_local.get("gt_label","") == "BL")).sum()) if "gt_label" in df_ctrl_local.columns else np.nan
    fp_moves_per_min = (fp_moves / total_min) if (total_min > 0 and not np.isnan(fp_moves)) else np.nan
    mask_w = df_ctrl_local["state"] == "WALKING"
    walk_time_s = float((to_num(df_ctrl_local.loc[mask_w, "t_end"]) - to_num(df_ctrl_local.loc[mask_w, "t_start"])).fillna(0).sum())
    return df_ctrl_local, {
        "walk_time_s": walk_time_s,
        "moves": moves,
        "stops": stops,
        "fp_moves": fp_moves,
        "fp_moves_per_min": fp_moves_per_min
    }, params

# --- Define la rejilla de búsqueda (cauto pero útil) ---
PARAM_GRID = {
    "mi_start_hi":  [0.68, 0.70, 0.72],
    "mi_start_lo":  [0.56, 0.58, 0.60, 0.62],
    "att_start_lo": [0.38, 0.40, 0.45],
    "mi_stop":      [0.38, 0.40, 0.42],
    "require_att_on_high": [True, False],
    # fijos:
    "index_window_s": [PAPER_PARAMS.get("index_window_s", 5.0)],  # usamos tu ventana actual (5 s)
    "cooldown_s":     [PAPER_PARAMS.get("cooldown_s", 5.0)],
    "initial_state":  [PAPER_PARAMS.get("initial_state", "STANDING")],
}

def param_combos(grid):
    keys = list(grid.keys())
    lists = [grid[k] for k in keys]
    for values in itertools.product(*lists):
        combo = dict(zip(keys, values))
        # evitar incoherencias: mi_start_hi >= mi_start_lo
        if combo["mi_start_hi"] < combo["mi_start_lo"]:
            continue
        yield combo

# --- Ejecutar grid ---
rows = []
ctrl_cache = {}  # opcional: cache si quieres inspeccionar luego
for combo in param_combos(PARAM_GRID):
    # separa overrides de los fijos
    overrides = {k: v for k, v in combo.items() if k not in ("index_window_s","cooldown_s","initial_state")}
    fixed = {k: v for k, v in combo.items() if k in ("index_window_s","cooldown_s","initial_state")}
    _, metrics, used_params = run_policy_with(
        df_out,
        base_params={**PAPER_PARAMS, **fixed},
        **overrides
    )
    rows.append({**metrics, **used_params})

results_df = pd.DataFrame(rows)

# --- Selección del mejor bajo restricción FP_MOVE/min ≤ 1.0 ---
MAX_FP_PER_MIN = 1.0
feasible = results_df[np.isfinite(results_df["fp_moves_per_min"]) & (results_df["fp_moves_per_min"] <= MAX_FP_PER_MIN)].copy()
if feasible.empty:
    # Si no hay factibles, coge el mínimo FP/min y luego maximiza walk_time
    best = (results_df
            .sort_values(["fp_moves_per_min", "walk_time_s"], ascending=[True, False])
            .iloc[0])
else:
    # Entre los factibles: maximiza walking time, desempata por FP/min y luego por menos MOVEs
    best = (feasible
            .sort_values(["walk_time_s", "fp_moves_per_min", "moves"], ascending=[False, True, True])
            .iloc[0])

# --- Mostrar Top-10 para inspección ---
def _fmt(df):
    cols = ["walk_time_s","fp_moves_per_min","fp_moves","moves","stops",
            "mi_start_hi","mi_start_lo","att_start_lo","mi_stop","require_att_on_high"]
    return (df[cols]
            .assign(walk_time_s=lambda d: d["walk_time_s"].round(1),
                    fp_moves_per_min=lambda d: d["fp_moves_per_min"].round(3))
            )

print("\n=== GRID-SEARCH RESULTS (Top-10 by objective) ===")
if feasible.empty:
    top = results_df.sort_values(["fp_moves_per_min","walk_time_s"], ascending=[True, False]).head(10)
else:
    top = feasible.sort_values(["walk_time_s","fp_moves_per_min","moves"], ascending=[False, True, True]).head(10)
print(_fmt(top))

print("\n=== BEST SETTING ===")
print(_fmt(pd.DataFrame([best])))

# --- Ejecutar y exponer DF control con el mejor set ---
best_params = {k: best[k] for k in best.index if k in PAPER_PARAMS or k in ["require_att_on_high","mi_start_hi","mi_start_lo","att_start_lo","mi_stop"]}
# Asegurar confirmación = 1
best_params["consecutive_start_needed"] = 1
with suppress_stdout():
    df_ctrl_best = apply_paper_control_policy(df_out, {**PAPER_PARAMS, **best_params})

print("\nPreview (best setting):")
print(df_ctrl_best[["file","t_start","t_end","p_MI","p_ATT","MI_index","ATT_index","state","command","cooldown_active"]].head(15))

# Mostrar parámetros exactos del mejor set (clave por clave)
print("\nBEST PARAMS (full):")
for k in ["mi_start_hi","mi_start_lo","att_start_lo","mi_stop","require_att_on_high",
          "index_window_s","cooldown_s","initial_state","consecutive_start_needed"]:
    if k in best.index:
        print(f"  {k}: {best[k]}")
    elif k in PAPER_PARAMS:
        print(f"  {k}: {PAPER_PARAMS[k]}")

# Aplicar best params a PAPER_PARAMS (para usar de ahora en adelante)
PAPER_PARAMS.update({
    "mi_start_hi": float(best["mi_start_hi"]),
    "mi_start_lo": float(best["mi_start_lo"]),
    "att_start_lo": float(best["att_start_lo"]),
    "mi_stop": float(best["mi_stop"]),
    "require_att_on_high": bool(best["require_att_on_high"]),
    "consecutive_start_needed": 1,  # tu requisito
})
print("\nPAPER_PARAMS updated with best setting.")




