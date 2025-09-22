import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from bson import loads
import re
import numpy as np
import pandas as pd

# -------------------- Configuración de segmentos y épocas (igual en ambos modelos) --------------------
SEGMENTS_S = [(5,125), (130,250), (255,375), (380,500)]
LABELS_MI  = ['MI', 'BL', 'MI', 'BL']       # MotorModel.py
LABELS_ATT = ['ATT','REST','ATT','REST']    # AttentionModel.py
EPOCH_LEN_S = 5

# -------------------- Carga y epochado (idéntico a tus scripts) --------------------
def load_all_bson(folder, pattern="*.rec.bson"):
    folder = Path(folder)
    docs = []
    for fp in sorted(folder.glob(pattern)):
        try:
            with open(fp, "rb") as f:
                doc = loads(f.read())
            docs.append((doc, fp.name))
        except Exception as e:
            print(f"[WARN] Could not read {fp.name}: {e}")
    print(f"Loaded {len(docs)} BSON files.")
    return docs

def ensure_time_first(arr):
    arr = np.asarray(arr)
    if arr.ndim != 2: raise ValueError("EEG must be 2D.")
    if arr.shape[0] < arr.shape[1]:
        return arr.T
    return arr

def epochar_segment(seg_data, fs, epoch_len_s=5):
    seg_data = ensure_time_first(seg_data)
    L = int(round(epoch_len_s * fs))
    n = seg_data.shape[0] // L
    if n == 0:
        return np.empty((0, L, seg_data.shape[1])), np.empty((0,))
    seg_data = seg_data[:n*L, :]
    epochs = seg_data.reshape(n, L, seg_data.shape[1])
    epoch_starts_s = np.arange(n) * epoch_len_s
    return epochs, epoch_starts_s

def process_doc_both_labels(doc, source_name,
                            segments_s=SEGMENTS_S,
                            labels_mi=LABELS_MI,
                            labels_att=LABELS_ATT,
                            epoch_len_s=EPOCH_LEN_S):
    eeg = doc["eeg"]
    fs = float(eeg["fs"])
    signal = ensure_time_first(eeg["signal"])
    n_channels = signal.shape[1]

    X_list, y_mi_list, y_att_list, meta_rows = [], [], [], []
    for seg_idx, (seg, lab_mi, lab_att) in enumerate(zip(segments_s, labels_mi, labels_att), start=1):
        a, b = seg
        i0, i1 = int(round(a*fs)), int(round(b*fs))
        seg_sig = signal[i0:i1, :]
        ep, ep_starts = epochar_segment(seg_sig, fs, epoch_len_s)
        if ep.size == 0:
            continue
        for k in range(ep.shape[0]):
            meta_rows.append({
                "file": source_name,
                "trial_idx": seg_idx,
                "epoch_idx": k,
                "epoch_start_s": float(a + ep_starts[k]),
                "epoch_end_s": float(a + ep_starts[k] + epoch_len_s),
                "fs": fs,
                "n_channels": n_channels
            })
        X_list.append(ep)
        y_mi_list.extend([lab_mi]  * ep.shape[0])
        y_att_list.extend([lab_att] * ep.shape[0])

    if not X_list:
        return np.empty((0, int(round(epoch_len_s*fs)), 0)), np.empty((0,)), np.empty((0,)), pd.DataFrame()

    X = np.concatenate(X_list, axis=0)
    y_mi = np.array(y_mi_list, dtype=object)
    y_att = np.array(y_att_list, dtype=object)
    meta = pd.DataFrame(meta_rows)
    return X, y_mi, y_att, meta

def process_folder_both_labels(folder, pattern="*.rec.bson", epoch_len_s=EPOCH_LEN_S):
    docs = load_all_bson(folder, pattern)
    X_all, y_mi_all, y_att_all, metas = [], [], [], []
    for doc, name in docs:
        X, y_mi, y_att, meta = process_doc_both_labels(doc, name, epoch_len_s=epoch_len_s)
        if X.size:
            X_all.append(X); y_mi_all.append(y_mi); y_att_all.append(y_att); metas.append(meta)
    if not X_all:
        print("No epochs.")
        return np.empty((0,)), np.empty((0,)), np.empty((0,)), pd.DataFrame()
    X_all = np.concatenate(X_all, axis=0)
    y_mi_all = np.concatenate(y_mi_all, axis=0)
    y_att_all = np.concatenate(y_att_all, axis=0)
    meta_all = pd.concat(metas, ignore_index=True)
    return X_all, y_mi_all, y_att_all, meta_all

# -------------------- Preprocesado y features (tomados de tus modelos) --------------------
from scipy.signal import butter, filtfilt, iirnotch, welch
from numpy.linalg import eigh

def demean(x): return x - np.mean(x, axis=0, keepdims=True)
def notch(x, fs, f0=50.0, Q=35.0):
    b, a = iirnotch(f0/(fs/2), Q); return filtfilt(b, a, x, axis=0)
def bandpass(x, fs, fmin, fmax, order=4):
    b, a = butter(order, [fmin/(fs/2), fmax/(fs/2)], btype='band'); return filtfilt(b, a, x, axis=0)
def car(x): return x - np.mean(x, axis=1, keepdims=True)

# --- Attention features (frontal θ/α/β + ratios), como en AttentionModel.py ---
FRONTAL = ['F3','Fz','F4']
BANDS_ATT = {'theta': (4,8), 'alpha': (8,13), 'betaL': (13,30)}
WIN_S = 2.0
OVERLAP = 0.5

def band_logpower(sig, fs, fmin, fmax, win_s=WIN_S, overlap=OVERLAP):
    nperseg = int(round(win_s*fs)); noverlap = int(round(nperseg*overlap))
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    m = (f >= fmin) & (f <= fmax)
    return np.log(np.trapz(Pxx[m], f[m]) + 1e-12)

def extract_attention_features(epoch, fs, ch_names):
    ep = car(epoch)
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
        feats += [th, al, be, th-al, th-be, be - np.log(np.exp(al)+np.exp(th)+1e-12)]
    idxs = [ch2idx[ch] for ch in have]
    sig_avg = np.mean(ep[:, idxs], axis=1)
    th = band_logpower(sig_avg, fs, *BANDS_ATT['theta'])
    al = band_logpower(sig_avg, fs, *BANDS_ATT['alpha'])
    be = band_logpower(sig_avg, fs, *BANDS_ATT['betaL'])
    feats += [th, al, be, th-al, th-be, be - np.log(np.exp(al)+np.exp(th)+1e-12)]
    return np.array(feats, dtype=float)

# --- Motor features (tangent space + LDA), como en MotorModel.py ---
def cov_spd(epoch, ridge=1e-6):
    C = np.cov(epoch.T); C = C + ridge*np.eye(C.shape[0])
    lam, V = eigh(C); lam = np.clip(lam, 1e-10, None)
    return V @ np.diag(lam) @ V.T

def spd_log(C):
    lam, V = eigh(C); lam = np.clip(lam, 1e-10, None)
    return V @ np.diag(np.log(lam)) @ V.T

def vec_sym(M):
    C = M.shape[0]
    idx_triu = np.triu_indices(C)
    v = M[idx_triu].astype(float)
    off = idx_triu[0] != idx_triu[1]
    v[off] *= np.sqrt(2.0)
    return v

BANDS_TS = [(8,12), (13,30), (8,30)]
RIDGE_COV = 1e-6

def make_ts_fit_transform(epochs, fs, bands=BANDS_TS, ridge=RIDGE_COV):
    def _fit_transform(idxs):
        X_bands_tr = []
        L_refs = []
        for (fmin, fmax) in bands:
            Xb = np.array([bandpass(epochs[i], fs, fmin, fmax, 4) for i in idxs])
            C_tr = np.array([cov_spd(Xb[i], ridge=ridge) for i in range(len(idxs))])
            # referencia log-Euclidiana
            logs = np.array([spd_log(C) for C in C_tr])
            L_ref = np.mean(logs, axis=0)     # log-mean
            L_refs.append(L_ref)
            Xtr_tan = np.vstack([vec_sym(spd_log(C) - L_ref) for C in C_tr])
            X_bands_tr.append(Xtr_tan)
        return np.hstack(X_bands_tr), L_refs
    def _transform(idxs, L_refs):
        X_bands_te = []
        k = 0
        for (fmin, fmax) in bands:
            Xb = np.array([bandpass(epochs[i], fs, fmin, fmax, 4) for i in idxs])
            C_te = np.array([cov_spd(Xb[i], ridge=ridge) for i in range(len(idxs))])
            L_ref = L_refs[k]; k += 1
            Xte_tan = np.vstack([vec_sym(spd_log(C) - L_ref) for C in C_te])
            X_bands_te.append(Xte_tan)
        return np.hstack(X_bands_te)
    return _fit_transform, _transform

# -------------------- Entrenamiento / evaluación cruzada dual --------------------
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

# --- New thresholds ---
TH_MI_STRONG = 0.68   # Regla A
TH_MI_WEAK   = 0.56   # parte baja de Regla B
TH_ATT_HELP  = 0.38   # atención mínima en Regla B

def dual_score(mi_p, att_p):
    """
    Continuous score consistent with the new rules:
      - if p_MI >= TH_MI_STRONG: score = p_MI
      - if TH_MI_WEAK <= p_MI < TH_MI_STRONG:
            score grows linearly from TH_MI_WEAK -> TH_MI_STRONG
            as ATT goes from 0 -> TH_ATT_HELP (clipped)
      - if p_MI < TH_MI_WEAK: score = p_MI
    """
    mi = np.asarray(mi_p); att = np.asarray(att_p)
    score = mi.copy()

    mid = (mi >= TH_MI_WEAK) & (mi < TH_MI_STRONG)
    # length of the "gap" to fill with attention help
    gap = (TH_MI_STRONG - TH_MI_WEAK)
    score[mid] = TH_MI_WEAK + gap * np.minimum(1.0, att[mid] / TH_ATT_HELP)

    return np.clip(score, 0.0, 1.0)

def dual_predict(mi_p, att_p):
    mi = np.asarray(mi_p); att = np.asarray(att_p)
    return ((mi >= TH_MI_STRONG) | ((mi >= TH_MI_WEAK) & (att >= TH_ATT_HELP))).astype(int)

def evaluate_dual(X, y_mi, y_att, meta, n_splits=10, test_size=0.25,
                  csv_metrics="metrics_dual_splits.csv",
                  csv_preds="preds_dual_all.csv"):
    # Ground-truth dual: positivo si MI (por tu premisa: MI ⊆ ATT)
    y_true = (y_mi == 'MI').astype(int)
    groups = meta["file"].values

    # fs y nombres de canales
    fs_unique = meta["fs"].unique()
    if len(fs_unique) != 1:
        raise ValueError("Expected a single sampling rate.")
    fs = float(fs_unique[0])
    assumed = ['F3','C3','P3','Fz','Cz','F4','C4','P4']
    ch_names = assumed if X.shape[2] == len(assumed) else [f"CH{i}" for i in range(X.shape[2])]

    # Preprocesado base para epochs (común a ambos modelos)
    X_pre = []
    for i in range(X.shape[0]):
        ep = demean(X[i]); ep = notch(ep, fs, 50.0, 35.0); ep = bandpass(ep, fs, 0.5, 40.0, 4); ep = car(ep)
        X_pre.append(ep)
    X_pre = np.asarray(X_pre)

    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

    rows = []
    cms = []
    all_rows_pred = []

    fit_transform_ts, transform_ts = make_ts_fit_transform(X_pre, fs, BANDS_TS, RIDGE_COV)

    for k, (tr, te) in enumerate(gss.split(X_pre, y_true, groups), 1):
        # ---------- Modelo MI (TS + LDA) ----------
        Xtr_tan, L_refs = fit_transform_ts(tr)
        Xte_tan = transform_ts(te, L_refs)

        sc_mi = StandardScaler().fit(Xtr_tan)
        Xtr_mi = sc_mi.transform(Xtr_tan); Xte_mi = sc_mi.transform(Xte_tan)

        clf_mi = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf_mi.fit(Xtr_mi, (y_mi[tr]=='MI').astype(int))
        p_mi_te = clf_mi.predict_proba(Xte_mi)[:,1]

        # ---------- Modelo ATT (frontal features + LDA) ----------
        # build features
        Xtr_att_feats = np.vstack([extract_attention_features(X_pre[i], fs, ch_names) for i in tr])
        Xte_att_feats = np.vstack([extract_attention_features(X_pre[i], fs, ch_names) for i in te])

        sc_att = StandardScaler().fit(Xtr_att_feats)
        Xtr_att = sc_att.transform(Xtr_att_feats); Xte_att = sc_att.transform(Xte_att_feats)

        clf_att = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf_att.fit(Xtr_att, (y_att[tr]=='ATT').astype(int))
        p_att_te = clf_att.predict_proba(Xte_att)[:,1]

        # ---------- Clasificación dual ----------
        y_true_te = y_true[te]
        y_pred_dual = dual_predict(p_mi_te, p_att_te)
        scores_dual = dual_score(p_mi_te, p_att_te)

        # ---------- Métricas ----------
        cm = confusion_matrix(y_true_te, y_pred_dual, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_true_te, y_pred_dual)
        try:
            auc = roc_auc_score(y_true_te, scores_dual)
        except ValueError:
            auc = np.nan
        sens = recall_score(y_true_te, y_pred_dual, pos_label=1, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        f1   = f1_score(y_true_te, y_pred_dual, pos_label=1, zero_division=0)

        rows.append({
            "split": k, "n_test": len(te),
            "ACC": acc, "AUC": auc, "Sensitivity": sens, "Specificity": spec, "F1": f1,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        })
        cms.append(cm)

        # guardar predicciones por época
        df_te = pd.DataFrame({
            "split": k,
            "file": meta["file"].values[te],
            "epoch_idx": meta["epoch_idx"].values[te],
            "y_true": y_true_te,
            "p_MI": p_mi_te,
            "p_ATT": p_att_te,
            "dual_score": scores_dual,
            "y_pred_dual": y_pred_dual
        })
        all_rows_pred.append(df_te)

        print(f"[Dual split {k:02d}] ACC={acc:.3f} AUC={auc:.3f} Sens={sens:.3f} Spec={spec:.3f} F1={f1:.3f} n_test={len(te)}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(csv_metrics, index=False)

    preds_df = pd.concat(all_rows_pred, ignore_index=True)
    preds_df.to_csv(csv_preds, index=False)

    # Resumen (media ± std)
    def ms(x): return f"{np.nanmean(x):.3f} ± {np.nanstd(x):.3f}"
    print("\n==== DUAL SUMMARY over splits ====")
    for key in ["ACC","AUC","Sensitivity","Specificity","F1"]:
        print(f"{key}: {ms(metrics_df[key].values)}")

    # Matriz de confusión agregada (conteos)
    cm_sum = np.sum(np.stack(cms, axis=0), axis=0)
    print("\nAggregated confusion matrix (counts):")
    print(cm_sum)

    return metrics_df, preds_df, cm_sum


import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, labels=("BL","MI"), save=True, path="figs/confusion_matrix.png"):
    import os, numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_perc = cm.astype(float) / row_sums

    # anotaciones: conteo + porcentaje por fila
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i,j]}\n({cm_perc[i,j]:.1%})"

    plt.figure(figsize=(6,5))
    ax = sns.heatmap(
        cm_perc, annot=annot, fmt="", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        vmin=0, vmax=1, cbar=True
    )

    # Colorbar 0–100%
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.linspace(0, 1, 5))                 # 0,20,40,60,80,100
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    ax.set_ylabel("Actual value")
    ax.set_xlabel("Predicted value")
    ax.set_title("Confusion matrix")

    if save:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()

# -------------------- MAIN --------------------
if __name__ == "__main__":
    # Cambia esta ruta a tu carpeta con *.rec.bson
    folder = r"C:\Users\SGP02\MEDUSA\v2024\data"
    X, y_mi, y_att, meta = process_folder_both_labels(folder, pattern="*.rec.bson", epoch_len_s=EPOCH_LEN_S)

    print("X:", X.shape, "(N_ep, L, C)")
    print("y_mi uniques:", np.unique(y_mi), " | y_att uniques:", np.unique(y_att))
    print("meta rows:", len(meta))

    metrics_df, preds_df, cm_sum = evaluate_dual(
        X, y_mi, y_att, meta,
        n_splits=10, test_size=0.25,
        csv_metrics="metrics_dual_splits.csv",
        csv_preds="preds_dual_all.csv"
    )

    plot_confusion_matrix(cm_sum, labels=["BL", "MI+ATT"])

    print("\nPer-split metrics saved to: metrics_dual_splits.csv")
    print("Per-epoch predictions saved to: preds_dual_all.csv")
