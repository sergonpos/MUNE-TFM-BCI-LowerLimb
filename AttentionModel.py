
# docs = load_all_bson(r"C:\Users\SGP02\MEDUSA\v2024\data")

from pathlib import Path
from bson import loads
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# --- Configuración de tus tramos y etiquetas ---
SEGMENTS_S = [(5,125), (130,250), (255,375), (380,500)]
LABELS     = ['ATT',     'REST',      'ATT',      'REST'     ]
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


# =================== PIPELINE ATENCIÓN (CAR + frontal θ/α/β + ratios) ===================
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from scipy.signal import iirnotch, butter, filtfilt, welch
import numpy as np

# --- Config ---
ATT_LABEL = 'ATT'   # cambia a tus etiquetas reales; para probar: ATT_LABEL='MI', NONATT_LABEL='BL'
NONATT_LABEL = 'REST'
# Para probar sin nuevo etiquetado:
# ATT_LABEL, NONATT_LABEL = 'MI', 'BL'

FRONTAL = ['F3','Fz','F4']
BANDS = {'theta': (4,8), 'alpha': (8,13), 'betaL': (13,30)}
WIN_S = 2.0     # ventana de Welch (s) -> si tus épocas son de 3 s, WIN_S=2 va bien
OVERLAP = 0.5
N_SPLITS = 10
TEST_SIZE = 0.25

# --- utilidades ---
def demean(x): return x - np.mean(x, axis=0, keepdims=True)
def apply_notch(x, fs, f0=50.0, Q=35.0):
    b, a = iirnotch(f0/(fs/2), Q); return filtfilt(b, a, x, axis=0)
def apply_bandpass(x, fs, fmin=0.5, fmax=30.0, order=4):
    b, a = butter(order, [fmin/(fs/2), fmax/(fs/2)], btype='band'); return filtfilt(b, a, x, axis=0)
def apply_car(x):  # x: (L, C)
    return x - np.mean(x, axis=1, keepdims=True)

def band_logpower(sig, fs, fmin, fmax, win_s=WIN_S, overlap=OVERLAP):
    nperseg = int(round(win_s*fs)); noverlap = int(round(nperseg*overlap))
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    m = (f >= fmin) & (f <= fmax)
    return np.log(np.trapz(Pxx[m], f[m]) + 1e-12)

def extract_attention_features(epoch, fs, ch_names):
    """
    epoch: (L, C), ya filtrado 0.5-40 y con notch; este bloque aplica CAR.
    features: log-powers theta/alpha/beta en F3/Fz/F4 + ratios + frontal-average
    """
    # referencia CAR
    ep = apply_car(epoch)

    # índices de canales
    ch2idx = {ch:i for i, ch in enumerate(ch_names)}
    have = [ch for ch in FRONTAL if ch in ch2idx]
    if not have:  # si faltan nombres, usa los 3 primeros canales como frontal
        have = [ch_names[i] for i in range(min(3, len(ch_names)))]
        ch2idx = {ch:i for i, ch in enumerate(ch_names)}

    feats = []
    # por canal frontal
    for ch in have:
        sig = ep[:, ch2idx[ch]]
        th = band_logpower(sig, fs, *BANDS['theta'])
        al = band_logpower(sig, fs, *BANDS['alpha'])
        be = band_logpower(sig, fs, *BANDS['betaL'])
        # ratios
        th_al = th - al                 # log(θ/α) = logθ - logα
        th_be = th - be                 # log(θ/β)
        ei    = be - np.log(np.exp(al) + np.exp(th) + 1e-12)  # log[β/(α+θ)]
        feats += [th, al, be, th_al, th_be, ei]

    # frontal average (robustez)
    idxs = [ch2idx[ch] for ch in have]
    sig_avg = np.mean(ep[:, idxs], axis=1)
    th = band_logpower(sig_avg, fs, *BANDS['theta'])
    al = band_logpower(sig_avg, fs, *BANDS['alpha'])
    be = band_logpower(sig_avg, fs, *BANDS['betaL'])
    th_al = th - al
    th_be = th - be
    ei    = be - np.log(np.exp(al) + np.exp(th) + 1e-12)
    feats += [th, al, be, th_al, th_be, ei]

    return np.array(feats, dtype=float)

# --- preparar etiquetas ATT vs REST ---
fs_unique = meta["fs"].unique()
if len(fs_unique) != 1: raise ValueError("Se espera una única fs.")
fs = float(fs_unique[0])

# nombres de canales (si no los tienes, asume orden conocido)
assumed = ['F3','C3','P3','Fz','Cz','F4','C4','P4']
ch_names = assumed if X.shape[2]==len(assumed) else [f"CH{i}" for i in range(X.shape[2])]

mask = np.isin(y, [ATT_LABEL, NONATT_LABEL])
if not mask.any():
    raise RuntimeError("No hay etiquetas ATT/REST en y. Ajusta ATT_LABEL/NONATT_LABEL o crea etiquetas para atención.")
X_att, y_att, meta_att = X[mask], y[mask], meta[mask]
y_bin = (y_att == ATT_LABEL).astype(int)
groups = meta_att["file"].values

# --- preprocesado + features ---
X_feats = []
for i in range(X_att.shape[0]):
    ep = demean(X_att[i])
    ep = apply_notch(ep, fs, 50.0, 35.0)
    ep = apply_bandpass(ep, fs, 0.5, 40.0, 4)
    feats = extract_attention_features(ep, fs, ch_names)
    X_feats.append(feats)
X_feats = np.asarray(X_feats, dtype=float)
print("Attention features shape:", X_feats.shape)

# --- evaluación multi-split 10x ---
gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
accs, aucs = [], []
for k, (tr, te) in enumerate(gss.split(X_feats, y_bin, groups), 1):
    scaler = StandardScaler().fit(X_feats[tr])
    Xtr = scaler.transform(X_feats[tr]); Xte = scaler.transform(X_feats[te])
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xtr, y_bin[tr])
    proba = clf.predict_proba(Xte)[:,1]; pred = (proba>=0.5).astype(int)
    acc = accuracy_score(y_bin[te], pred)
    try: auc = roc_auc_score(y_bin[te], proba)
    except: auc = np.nan
    accs.append(acc); aucs.append(auc)
    print(f"[ATT Split {k:02d}] ACC={acc:.3f}  AUC={auc:.3f}  n_test={len(te)}")

print(f"\nATT resumen 10x: ACC={np.nanmean(accs):.3f} ± {np.nanstd(accs):.3f} | AUC={np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}")


# =================== ENTRENAR MODELO FINAL (FULL) Y GUARDARLO ===================
from joblib import dump
import json
from datetime import datetime

# 1) Reajusta scaler en TODO el conjunto y entrena LDA en TODO (para desplegar)
scaler_full = StandardScaler().fit(X_feats)
X_all_sc = scaler_full.transform(X_feats)

lda_full = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
lda_full.fit(X_all_sc, y_bin)

# 2) Guarda modelo y scaler
MODEL_OUT  = "model_att_frontal_lda_full.joblib"
SCALER_OUT = "scaler_att_frontal_full.joblib"
dump(lda_full, MODEL_OUT)
dump(scaler_full, SCALER_OUT)

# 3) (Recomendado) Guarda metadatos de extracción de features para reproducibilidad
meta_att_conf = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "fs": fs,
    "channels_expected": ch_names,          # orden usado al extraer features
    "reference": "CAR",
    "bands": {"theta":[4,8], "alpha":[8,13], "betaL":[13,30]},
    "welch": {"win_s": WIN_S, "overlap": OVERLAP},
    "features": [
        "Per-channel: [logP_theta, logP_alpha, logP_betaL, log(theta/alpha), log(theta/betaL), EI]",
        "Frontal-avg: same 6 features"
    ],
    "frontal_set": FRONTAL,
    "label_pos": ATT_LABEL,                 # clase positiva (1)
    "label_neg": NONATT_LABEL               # clase negativa (0)
}
with open("att_feature_config.json", "w", encoding="utf-8") as f:
    json.dump(meta_att_conf, f, indent=2, ensure_ascii=False)

print(f"\nGuardados:\n - {MODEL_OUT}\n - {SCALER_OUT}\n - att_feature_config.json")


from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, classification_report,
    roc_curve, auc, average_precision_score, precision_recall_curve
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# containers
metrics_rows = []
cms = []               # raw confusion matrices per split (2x2)
rocs = []              # (fpr, tpr, auc) per split
prs  = []              # (precision, recall, ap) per split
y_all, proba_all = [], []  # aggregate (across test folds) for macro summaries
Path("figs").mkdir(exist_ok=True)

gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
for k, (tr, te) in enumerate(gss.split(X_feats, y_bin, groups), 1):
    scaler = StandardScaler().fit(X_feats[tr])
    Xtr = scaler.transform(X_feats[tr])
    Xte = scaler.transform(X_feats[te])

    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xtr, y_bin[tr])
    proba = clf.predict_proba(Xte)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    y_true = y_bin[te]

    # confusion matrix
    cm = confusion_matrix(y_true, pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    cms.append(cm)

    # metrics
    acc  = (tp + tn) / (tp + tn + fp + fn)
    bac  = balanced_accuracy_score(y_true, pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan   # recall of positive
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    prec = precision_score(y_true, pred, zero_division=0)
    rec  = recall_score(y_true, pred, zero_division=0)
    f1   = f1_score(y_true, pred, zero_division=0)
    mcc  = matthews_corrcoef(y_true, pred) if len(np.unique(y_true)) > 1 else np.nan

    # ROC & PR
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    pr_prec, pr_rec, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)

    rocs.append((fpr, tpr, roc_auc))
    prs.append((pr_prec, pr_rec, ap))

    metrics_rows.append({
        "split": k, "n_test": len(te),
        "ACC": acc, "AUC": roc_auc, "AP": ap,
        "BalancedACC": bac, "Precision": prec, "Recall": rec, "F1": f1,
        "Sensitivity": sens, "Specificity": spec, "MCC": mcc,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    })

    # accumulate for aggregate curves
    y_all.append(y_true); proba_all.append(proba)

metrics_df = pd.DataFrame(metrics_rows)
print("\nPer-split metrics:")
print(metrics_df.round(3))

# Save a CSV with all split metrics
metrics_df.to_csv("metrics_splits_attention.csv", index=False)

def mean_std_str(x):
    return f"{np.nanmean(x):.3f} ± {np.nanstd(x):.3f}"

print("\nSummary over splits:")
for key in ["ACC","AUC","AP","BalancedACC","Precision","Recall","F1","Sensitivity","Specificity","MCC"]:
    print(f"{key}: {mean_std_str(metrics_df[key].values)}")

# Average confusion matrix (raw and normalized)
cm_sum = np.sum(np.stack(cms, axis=0), axis=0)   # total counts across splits
cm_norm = cm_sum / cm_sum.sum(axis=1, keepdims=True)

print("\nAggregated confusion matrix (counts):")
print(cm_sum)
print("\nAggregated confusion matrix (row-normalized):")
print(np.round(cm_norm, 3))

# Aggregate for overall curves (note: samples can repeat across splits with GroupShuffleSplit)
y_all = np.concatenate(y_all)
proba_all = np.concatenate(proba_all)


import seaborn as sns
def plot_cm_combined(cm_counts, class_names=("REST(0)", "ATT(1)"),
                     normalize="row", out_path="figs/cm_combined_attention.png",
                     cmap="Blues"):
    """
    cm_counts: 2x2 (o NxN) confusion matrix con conteos (TN/FP/FN/TP agregados).
    normalize: 'row' (por defecto), 'col' o 'all' para cómo calcular los porcentajes.
    """
    cm_counts = np.asarray(cm_counts, dtype=float)

    # --- normalización para porcentajes ---
    if normalize == "row":
        denom = cm_counts.sum(axis=1, keepdims=True)
    elif normalize == "col":
        denom = cm_counts.sum(axis=0, keepdims=True)
    elif normalize == "all":
        denom = np.array([[cm_counts.sum()]])
    else:
        raise ValueError("normalize must be 'row', 'col', or 'all'")

    # Evitar divisiones por 0
    denom[denom == 0] = 1.0
    cm_perc = cm_counts / denom

    # --- anotaciones: "conteo\n(XX.X%)" ---
    annot = np.empty_like(cm_counts, dtype=object)
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            annot[i, j] = f"{int(cm_counts[i, j])}\n({cm_perc[i, j]*100:.1f}%)"

    # --- plot: coloreamos por porcentaje, anotamos con ambos ---
    plt.figure(figsize=(5.2, 4.6))
    ax = sns.heatmap(cm_perc, annot=annot, fmt="",
                     cmap=cmap, vmin=0, vmax=1, cbar=True,
                     xticklabels=class_names, yticklabels=class_names,
                     linewidths=1, linecolor="white", square=True)

    ax.set_xlabel("Predicted value", fontsize=11)
    ax.set_ylabel("Actual value", fontsize=11)
    norm_label = {"row": "Row-normalized (per true class)",
                  "col": "Column-normalized (per predicted class)",
                  "all": "Normalized by total"}[normalize]
    ax.set_title("Confusion Matrix", fontsize=12)

    cbar = ax.collections[0].colorbar
    cbar.set_label("%", rotation=0, labelpad=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels([f"{t*100:.0f}%" for t in [0, .25, .5, .75, 1]])

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved combined confusion matrix to: {out_path}")

class_names = ["BL", "ATT"]
plot_cm_combined(cm_sum, class_names, normalize="row",
                 out_path="figs/cm_combined_attention.png")



