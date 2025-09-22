
# docs = load_all_bson(r"C:\Users\SGP02\MEDUSA\v2024\data")

from pathlib import Path
from bson import loads
import numpy as np
import pandas as pd
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

print("X:", X.shape, "(N_ep, L, C)")
print("y:", y.shape, " | únicas:", np.unique(y))
print("meta filas:", len(meta))
print(meta.head())

# =================== MI: RIEMANNIANO (Sólo Tangent + LDA, versión mejorada) ===================
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import butter, filtfilt, iirnotch
from joblib import dump
import re
import numpy as np

# ---------- Utilidades de preprocesado ----------
def demean(x):
    return x - np.mean(x, axis=0, keepdims=True)

def apply_notch(x, fs, f0=50.0, Q=35.0):
    b, a = iirnotch(w0=f0/(fs/2), Q=Q)
    return filtfilt(b, a, x, axis=0)

def bandpass(x, fs, fmin, fmax, order=4):
    b, a = butter(order, [fmin/(fs/2), fmax/(fs/2)], btype='band')
    return filtfilt(b, a, x, axis=0)

def apply_car(epoch):  # (L, C)
    return epoch - np.mean(epoch, axis=1, keepdims=True)

# ---------- SPD + Log-Euclid ----------
def cov_spd(epoch, ridge=1e-6):
    # cov + ridge*I para estabilidad; garantiza SPD antes de log
    C = np.cov(epoch.T)
    C = C + ridge * np.eye(C.shape[0])
    # clamp eigenvalues por seguridad
    lam, V = eigh(C)
    lam = np.clip(lam, 1e-10, None)
    return V @ np.diag(lam) @ V.T

def spd_log(C):
    lam, V = eigh(C)
    lam = np.clip(lam, 1e-10, None)
    return V @ np.diag(np.log(lam)) @ V.T

def log_euclidean_mean(Cs):
    logs = np.array([spd_log(C) for C in Cs])
    return exp_from_log(np.mean(logs, axis=0))

def exp_from_log(L):
    lam, V = eigh(L)
    return V @ np.diag(np.exp(lam)) @ V.T

def vec_sym(M):
    """Vectoriza matriz simétrica con ponderación √2 en off-diagonal (isometría Frobenius)."""
    C = M.shape[0]
    idx_triu = np.triu_indices(C)
    v = M[idx_triu].astype(float)
    # multiplicar off-diagonales por √2
    off = idx_triu[0] != idx_triu[1]
    v[off] *= np.sqrt(2.0)
    return v

# ---------- Preparación común ----------
fs_unique = meta["fs"].unique()
if len(fs_unique) != 1:
    raise ValueError("Se espera una única fs para todas las épocas.")
fs = float(fs_unique[0])

y_bin = (y == 'MI').astype(int)
groups_file = meta["file"].values  # agrupa por archivo/grabación

# Deriva un "group by day" a partir del nombre del archivo, p.ej. "01-08-2025_..." -> "01-08-2025"
def extract_day(fn):
    m = re.match(r"(\d{2}-\d{2}-\d{4})", str(fn))
    return m.group(1) if m else "unknown"
groups_day = np.array([extract_day(fn) for fn in groups_file])

# Preprocesado estable previo (independiente de bandas específicas)
X_pre = []
for i in range(X.shape[0]):
    ep = demean(X[i])
    ep = apply_notch(ep, fs, 50.0, 35.0)
    ep = bandpass(ep, fs, 0.5, 40.0, 4)
    ep = apply_car(ep)
    X_pre.append(ep)
X_pre = np.asarray(X_pre)  # (N, L, C)

# ---------- Tangent-space features por bandas y concatenación ----------
BANDS_TS = [(8,12), (13,30), (8,30)]  # μ, β y banda amplia
RIDGE_COV = 1e-6                       # regularización de covarianza
USE_SHRINK_LDA = True                  # LDA con shrinkage auto

def ts_features_from_epochs(epochs, fs, bands, ridge=1e-6):
    """
    Para un conjunto de épocas (N, L, C), genera features concatenando espacio tangente
    por cada banda en 'bands'. Devuelve (N, D_total) y objetos de referencia por banda.
    """
    N = epochs.shape[0]
    refs = []     # (por banda) matrices L_ref (log de la media)
    feats_all = []

    # Se calcularán referencias en entrenamiento; aquí solo estructura
    # Este helper devuelve funciones closure para fit/transform
    def _fit_transform(idxs):
        X_bands_ref = []
        L_refs = []
        X_bands_tr = []
        for (fmin, fmax) in bands:
            # banda
            Xb = np.array([bandpass(epochs[i], fs, fmin, fmax, 4) for i in idxs])
            # covs SPD
            C_tr = np.array([cov_spd(Xb[i], ridge=ridge) for i in range(len(idxs))])
            # referencia log-Euclidiana
            C_ref = log_euclidean_mean(C_tr)
            L_ref = spd_log(C_ref)
            L_refs.append(L_ref)
            # proyección de train
            Xtr_tan = np.vstack([vec_sym(spd_log(C) - L_ref) for C in C_tr])
            X_bands_ref.append((fmin, fmax, L_ref))
            X_bands_tr.append(Xtr_tan)
        # concatenar features de todas las bandas
        Xtr_concat = np.hstack(X_bands_tr)
        return Xtr_concat, L_refs

    def _transform(idxs, L_refs):
        X_bands_te = []
        k = 0
        for (fmin, fmax) in bands:
            Xb = np.array([bandpass(epochs[i], fs, fmin, fmax, 4) for i in idxs])
            C_te = np.array([cov_spd(Xb[i], ridge=ridge) for i in range(len(idxs))])
            L_ref = L_refs[k]; k += 1
            Xte_tan = np.vstack([vec_sym(spd_log(C) - L_ref) for C in C_te])
            X_bands_te.append(Xte_tan)
        Xte_concat = np.hstack(X_bands_te)
        return Xte_concat

    return _fit_transform, _transform

fit_transform_ts, transform_ts = ts_features_from_epochs(X_pre, fs, BANDS_TS, ridge=RIDGE_COV)

# ---------- Evaluación: GroupShuffleSplit por archivo ----------
def eval_tangent_lda_by_file(n_reps=10, test_size=0.25):
    accs, aucs = [], []
    gss = GroupShuffleSplit(n_splits=n_reps, test_size=test_size)
    for rep, (tr, te) in enumerate(gss.split(X_pre, y_bin, groups_file), 1):
        # Tangent-space con referencias del train (por banda)
        Xtr_tan, L_refs = fit_transform_ts(tr)
        Xte_tan = transform_ts(te, L_refs)

        scaler = StandardScaler().fit(Xtr_tan)
        Xtr_sc = scaler.transform(Xtr_tan)
        Xte_sc = scaler.transform(Xte_tan)

        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto' if USE_SHRINK_LDA else None)
        clf.fit(Xtr_sc, y_bin[tr])
        proba = clf.predict_proba(Xte_sc)[:, 1]
        pred = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_bin[te], pred)
        try:
            auc = roc_auc_score(y_bin[te], proba)
        except ValueError:
            auc = np.nan
        accs.append(acc); aucs.append(auc)
        print(f"[Tangent+LDA (by file) {rep:02d}] ACC={acc:.3f}  AUC={auc:.3f}  n_test={len(te)}  (files={len(np.unique(groups_file[te]))})")

    print(f"\nTangent+LDA (by file) {n_reps}x: ACC={np.nanmean(accs):.3f} ± {np.nanstd(accs):.3f} | "
          f"AUC={np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}")

# ---------- Evaluación 2: Leave-One-Day-Out con Alineación por Día (EA) + Threshold Tuning ----------
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import roc_curve
from numpy.linalg import eigh
import numpy as np

# --- helpers EA + threshold ---
def mat_inv_sqrt(C):
    lam, V = eigh(C)
    lam = np.clip(lam, 1e-10, None)
    return V @ np.diag(lam**-0.5) @ V.T

def mean_cov(epochs):
    # media euclídea de covarianzas SPD con ridge
    Cs = [cov_spd(ep, ridge=RIDGE_COV) for ep in epochs]
    return np.mean(Cs, axis=0)

def apply_ea(epoch, C_mean_inv_sqrt):
    # alineación a la derecha: (L,C) @ C_mean^{-1/2}
    return epoch @ C_mean_inv_sqrt

def tune_threshold(proba, y_true):
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    best = np.argmax(j)
    return float(thr[best])

# Ejecuta ambas evaluaciones (por archivo conservas tu función previa)
eval_tangent_lda_by_file(n_reps=10, test_size=0.25)

# ---------- Entrenamiento FULL (para despliegue) ----------
# Fit FULL sin EA (porque no hay "día" objetivo). Guardamos además un umbral por defecto via CV.
Xtr_tan_full, L_refs_full = fit_transform_ts(np.arange(X_pre.shape[0]))
scaler_full = StandardScaler().fit(Xtr_tan_full)
Xtr_sc_full = scaler_full.transform(Xtr_tan_full)
lda_full = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto' if USE_SHRINK_LDA else None)
lda_full.fit(Xtr_sc_full, y_bin)

# Umbral por defecto via CV en TODO (útil para despliegue offline)
skf_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
thr_full_list = []
for tr_in, va_in in skf_full.split(Xtr_sc_full, y_bin):
    clf_in = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto' if USE_SHRINK_LDA else None)
    clf_in.fit(Xtr_sc_full[tr_in], y_bin[tr_in])
    proba_va = clf_in.predict_proba(Xtr_sc_full[va_in])[:, 1]
    thr_full_list.append(tune_threshold(proba_va, y_bin[va_in]))
default_threshold = float(np.median(thr_full_list))

# Empaquetado para inference
riem_tan_lda_full = {
    "bands": BANDS_TS,
    "fs": fs,
    "ridge_cov": RIDGE_COV,
    "L_refs": L_refs_full,   # referencias por banda
    "scaler": scaler_full,
    "clf": lda_full,
    "default_threshold": default_threshold
}
dump(riem_tan_lda_full, "model_riem_tangent_lda_full.joblib")
print(f"Guardado: model_riem_tangent_lda_full.joblib  | default_threshold={default_threshold:.3f}")

# ---------- Helper opcional para inferencia en tiempo real ----------
def predict_proba_tangent_lda(epoch_2d, model):
    """
    epoch_2d: ndarray (L, C) cruda (sin preprocesado).
    Devuelve probabilidad de clase 'MI'.
    """
    fs = model["fs"]
    # preprocesado base
    ep = demean(epoch_2d)
    ep = apply_notch(ep, fs, 50.0, 35.0)
    ep = bandpass(ep, fs, 0.5, 40.0, 4)
    ep = apply_car(ep)

    # features multi-banda usando L_refs FULL
    feats = []
    for (fmin, fmax), L_ref in zip(model["bands"], model["L_refs"]):
        ep_b = bandpass(ep, fs, fmin, fmax, 4)
        C = cov_spd(ep_b, ridge=model["ridge_cov"])
        v = vec_sym(spd_log(C) - L_ref)
        feats.append(v)
    x = np.hstack(feats).reshape(1, -1)

    x_sc = model["scaler"].transform(x)
    proba = model["clf"].predict_proba(x_sc)[0, 1]
    return proba

# ==== Metrics + One Combined Confusion Matrix (counts + %) ====
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper to draw a single CM with "count\n(percentage)" in each cell ---
def plot_cm_combined(cm_counts, class_names=("BL(0)","MI(1)"),
                     normalize="row", out_path="figs/cm_combined_MI.png",
                     cmap="Blues"):
    """
    cm_counts: NxN confusion matrix (counts).
    normalize: 'row' (percentages per true class), 'col', or 'all'.
    """
    cm_counts = np.asarray(cm_counts, dtype=float)

    # Denominator for percentages
    if normalize == "row":
        denom = cm_counts.sum(axis=1, keepdims=True)
        norm_label = "Row-normalized (per true class)"
    elif normalize == "col":
        denom = cm_counts.sum(axis=0, keepdims=True)
        norm_label = "Column-normalized (per predicted class)"
    elif normalize == "all":
        denom = np.array([[cm_counts.sum()]])
        norm_label = "Normalized by total"
    else:
        raise ValueError("normalize must be 'row', 'col', or 'all'.")

    denom[denom == 0] = 1.0
    cm_perc = cm_counts / denom

    # Compose annotations: "count\n(XX.X%)"
    annot = np.empty_like(cm_counts, dtype=object)
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            annot[i, j] = f"{int(cm_counts[i, j])}\n({cm_perc[i, j]*100:.1f}%)"

    plt.figure(figsize=(5.4, 4.8))
    ax = sns.heatmap(cm_perc, annot=annot, fmt="",
                     cmap=cmap, vmin=0, vmax=1, cbar=True, square=True,
                     xticklabels=class_names, yticklabels=class_names,
                     linewidths=1, linecolor="white")

    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Actual value")
    ax.set_title("Confusion matrix", fontsize=12)

    cbar = ax.collections[0].colorbar
    cbar.set_label("%", rotation=0, labelpad=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels([f"{t*100:.0f}%" for t in [0, .25, .5, .75, 1]])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved combined confusion matrix to: {out_path}")

# --- Evaluation with the requested metrics ---
def eval_ts_lda_metrics(n_splits=10, test_size=0.25, csv_out="metrics_MI_splits.csv"):
    """
    Uses your existing fit_transform_ts / transform_ts on (X_pre, y_bin) grouped by groups_file.
    Reports per-split metrics and a combined confusion matrix (counts + %).
    """
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size)
    rows = []
    cms = []

    for k, (tr, te) in enumerate(gss.split(X_pre, y_bin, groups_file), 1):
        # Tangent-space features w/ train refs
        Xtr_tan, L_refs = fit_transform_ts(tr)
        Xte_tan = transform_ts(te, L_refs)

        # Scale + train LDA
        scaler = StandardScaler().fit(Xtr_tan)
        Xtr_sc = scaler.transform(Xtr_tan)
        Xte_sc = scaler.transform(Xte_tan)

        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(Xtr_sc, y_bin[tr])

        proba = clf.predict_proba(Xte_sc)[:, 1]
        pred = (proba >= 0.5).astype(int)
        y_true = y_bin[te]

        # Confusion matrix & counts
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cms.append(cm)

        # Metrics requested
        acc = accuracy_score(y_true, pred)
        try:
            auc = roc_auc_score(y_true, proba)
        except ValueError:
            auc = np.nan
        sens = recall_score(y_true, pred, pos_label=1, zero_division=0)  # TPR
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan              # TNR
        f1   = f1_score(y_true, pred, pos_label=1, zero_division=0)

        rows.append({
            "split": k,
            "n_test": len(te),
            "ACC": acc,
            "AUC": auc,
            "Sensitivity_TPR": sens,
            "Specificity_TNR": spec,
            "F1": f1,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        })

        print(f"[MI TS+LDA {k:02d}] ACC={acc:.3f}  AUC={auc:.3f}  "
              f"Sens={sens:.3f}  Spec={spec:.3f}  F1={f1:.3f}  n_test={len(te)}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(csv_out, index=False)
    print(f"\n[INFO] Per-split metrics saved to: {csv_out}\n")
    print("Per-split metrics (rounded):\n", metrics_df.round(3))

    # Summary (mean ± std) for the requested metrics
    def ms(x): return f"{np.nanmean(x):.3f} ± {np.nanstd(x):.3f}"
    print("\nSummary over splits:")
    for key in ["ACC", "AUC", "Sensitivity_TPR", "Specificity_TNR", "F1"]:
        print(f"{key}: {ms(metrics_df[key].values)}")

    # Combined CM across splits (sum of counts)
    cm_sum = np.sum(np.stack(cms, axis=0), axis=0)
    print("\nAggregated confusion matrix (counts):\n", cm_sum)

    # Save the combined CM plot (counts + row %)
    class_names = ["BL", "MI"]
    plot_cm_combined(cm_sum, class_names=class_names,
                     normalize="row", out_path="figs/cm_combined_MI.png")

    return metrics_df, cm_sum

# ---- Run it ----
metrics_df_mi, cm_sum_mi = eval_ts_lda_metrics(n_splits=10, test_size=0.25,
                                               csv_out="metrics_MI_splits.csv")


