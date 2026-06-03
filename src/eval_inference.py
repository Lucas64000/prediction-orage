"""eval_inference.py — Ré-inférence batchée et multi-GPU des modèles de fin d'alerte.

Produit, pour chaque modèle présent dans `models/`, un flux de prédictions au format
standard du jury (`PREDICTION_COLS`), prêt pour `gain_risk_eval`.

On **garde tout** : une prédiction est émise à *chaque* éclair de *chaque* alerte
(aucune alerte ni aucun événement n'est abandonné). Le contexte d'entrée du modèle à
l'instant t est la fenêtre causale des `MAX_SEQ_LEN` derniers éclairs se terminant à t,
réalignée à 0 — c'est la longueur de séquence maximale vue à l'entraînement (les
datasets capaient l'entrée à 200 événements, sans jamais écarter d'alerte).

Optimisation : les modèles sont **causaux** (Transformer à masque triangulaire, GRU).
Pour les `MAX_SEQ_LEN` premiers événements d'une alerte, un *seul* forward sur le
préfixe fournit la prédiction à toutes les positions (position t = info jusqu'à t) —
ce qui couvre en une passe la grande majorité des alertes (médiane ≈ 4 éclairs). Seule
la queue des alertes de plus de 200 éclairs (~6 %) passe par des fenêtres glissantes.

Multi-GPU : les sessions de chaque modèle sont shardées sur tous les `cuda:i`
disponibles, un thread worker par device (calqué sur `train_parallel`). Repli CPU.

Conventions de sortie des modèles :
  - minutes (`v2`, bayésien)  : `forward → (intensity, time_pred, …)`, temps en minutes.
  - log-space (`v3`, MoE)     : `forward → (intensity, mu, log_sigma, …)`, minutes = exp(mu) − 1.
Incertitude : MC Dropout (n_mc passes). Confiance = Φ((30 − μ)/σ) si σ disponible,
sinon `clip((30 − pred)/30, 0, 1)`.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

import torch  # noqa: E402

REPO_DIR = _SRC_DIR.parent
MODELS_DIR = REPO_DIR / "models"

MAX_SEQ_LEN = 200  # longueur de séquence max vue à l'entraînement (contexte d'entrée)
N_MC = 30  # passes Monte-Carlo Dropout pour les modèles à incertitude
POINT_BATCH = 64  # sessions/fenêtres par batch (modèles déterministes)
MC_CAP = 240  # plafond n_mc × items par batch (mémoire des modèles MC)
MAX_GAP_MIN = 30

PREDICTION_COLS = [
    "airport",
    "airport_alert_id",
    "prediction_date",
    "predicted_date_end_alert",
    "confidence",
]

# Features tabulaires (XGBoost AFT, BNN) — cf. bnn_model.FEATURE_COLS.
TABULAR_FEATURE_COLS = [
    "time_since_start", "time_since_last_cg",
    "n_lightnings_15m", "n_lightnings_30m", "n_ic_15m", "n_ic_30m", "ic_ratio_15m",
    "mean_dist_15m", "mean_dist_30m", "dist_trend",
    "mean_abs_amp_15m", "max_abs_amp_15m",
    "dist", "amplitude", "maxis", "icloud",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]

# Registre des checkpoints séquentiels.
#   prepare : "v2" (12 features) ou "spatial" (20 features)
#   kind    : "minutes" (sortie directe) ou "logspace" (exp(mu) − 1)
#   mc      : True si incertitude par MC Dropout
SEQ_REGISTRY = [
    {"name": "V2 GRU", "file": "neural_hawkes_gru_v2.pt", "prepare": "v2",
     "module": "neural_hawkes_v2", "cls": "NeuralHawkesGRUv2",
     "kwargs": dict(input_dim=12, hidden_dim=64, dropout=0.1),
     "kind": "minutes", "mc": False},
    {"name": "V2 Transformer", "file": "neural_hawkes_transformer.pt", "prepare": "v2",
     "module": "neural_hawkes_v2", "cls": "NeuralHawkesTransformer",
     "kwargs": dict(input_dim=12, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1),
     "kind": "minutes", "mc": False},
    {"name": "V2 Transformer-Large", "file": "neural_hawkes_transformer_large.pt", "prepare": "v2",
     "module": "neural_hawkes_v2", "cls": "NeuralHawkesTransformer",
     "kwargs": dict(input_dim=12, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.15),
     "kind": "minutes", "mc": False},
    {"name": "V3 GRU gaussien", "file": "hawkes_gru_gaussian.pt", "prepare": "v2",
     "module": "neural_hawkes_v3", "cls": "GaussianHawkesGRU",
     "kwargs": dict(input_dim=12, hidden_dim=64, dropout=0.15),
     "kind": "logspace", "mc": True},
    {"name": "V3 Transformer gaussien", "file": "hawkes_tf_gaussian.pt", "prepare": "v2",
     "module": "neural_hawkes_v3", "cls": "GaussianHawkesTransformer",
     "kwargs": dict(input_dim=12, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.15),
     "kind": "logspace", "mc": True},
    {"name": "V3 Transformer-Large gaussien", "file": "hawkes_tf_large_gaussian.pt", "prepare": "v2",
     "module": "neural_hawkes_v3", "cls": "GaussianHawkesTransformer",
     "kwargs": dict(input_dim=12, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.15),
     "kind": "logspace", "mc": True},
    {"name": "Bayes MC Dropout", "file": "bayesian_hawkes_mc.pt", "prepare": "v2",
     "module": "bayesian_hawkes", "cls": "MCDropoutHawkesTransformer",
     "kwargs": dict(input_dim=12, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.2),
     "kind": "minutes", "mc": True},
    {"name": "Bayes Variationnel", "file": "bayesian_hawkes_var.pt", "prepare": "v2",
     "module": "bayesian_hawkes", "cls": "VariationalHawkesTransformer",
     "kwargs": dict(input_dim=12, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1, prior_sigma=1.0),
     "kind": "minutes", "mc": True},
    {"name": "Spatial MoE", "file": "spatial_moe.pt", "prepare": "spatial",
     "module": "spatial_moe_model", "cls": "SpatialMoETransformer",
     "kwargs": dict(input_dim=20, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.15),
     "kind": "logspace", "mc": True},
]


# ─── Devices ──────────────────────────────────────────────────────────────────

def detect_devices() -> list[torch.device]:
    """Liste les GPU CUDA disponibles (un device par GPU), sinon `[cpu]`."""
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return [torch.device("cpu")]


# ─── Chargement / normalisation des données ───────────────────────────────────

def load_eval_alerts(csv_path: str | Path) -> pd.DataFrame:
    """Charge un CSV d'alertes (schéma jury) et le normalise pour l'inférence.

    Renomme `alert_id → airport_alert_id`, filtre aux éclairs en alerte
    (`dist ≤ 20 km`, repérés par `alert_id` non nul), parse les dates en UTC, force
    `icloud` booléen et ajoute un `lightning_id` séquentiel (requis par les features).
    """
    df = pd.read_csv(csv_path)
    if "airport_alert_id" not in df.columns and "alert_id" in df.columns:
        df = df.rename(columns={"alert_id": "airport_alert_id"})
    df = df[df["airport_alert_id"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
    df["icloud"] = df["icloud"].astype(bool)
    df = df.sort_values(["airport", "airport_alert_id", "date"]).reset_index(drop=True)
    df["lightning_id"] = np.arange(len(df))
    return df


# ─── Outils de batching ───────────────────────────────────────────────────────

def _pad_batch(items: list[dict], device: torch.device):
    """Empile et padde des fenêtres (clés `features` (L,D), `mtimes` (L,), `n_win`).

    Returns:
        (features, timestamps, delta_times, padding_mask) sur `device`.
    """
    batch_size = len(items)
    max_len = max(item["n_win"] for item in items)
    feat_dim = items[0]["features"].shape[1]

    feats = np.zeros((batch_size, max_len, feat_dim), dtype=np.float32)
    mtimes = np.zeros((batch_size, max_len), dtype=np.float32)
    deltas = np.zeros((batch_size, max_len), dtype=np.float32)
    padding_mask = np.ones((batch_size, max_len), dtype=bool)

    for b, item in enumerate(items):
        n = item["n_win"]
        feats[b, :n] = item["features"]
        mtimes[b, :n] = item["mtimes"]
        if n > 1:
            deltas[b, 1:n] = np.diff(item["mtimes"])
        padding_mask[b, :n] = False

    return (
        torch.from_numpy(feats).to(device),
        torch.from_numpy(mtimes).to(device),
        torch.from_numpy(deltas).unsqueeze(-1).to(device),
        torch.from_numpy(padding_mask).to(device),
    )


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _forward_batch(model, feats, mtimes, deltas, pmask, kind: str, mc: int):
    """Forward (éventuellement MC) → (mean_minutes, std|None) en (B, L) numpy.

    Pour `mc > 0` : batch répliqué `mc` fois (dropout actif), agrégé par position. En
    log-space, `std` combine l'épistémique (variance MC) et l'aléatorique (σ du modèle,
    méthode delta log→minutes). Pour `mc == 0` : passe unique déterministe, `std=None`.
    """
    with torch.no_grad():
        if mc > 0:
            batch_size = feats.shape[0]
            f = feats.repeat(mc, 1, 1)
            t = mtimes.repeat(mc, 1)
            d = deltas.repeat(mc, 1, 1)
            pm = pmask.repeat(mc, 1)
            out = model(f, t, d, pm)
            head = out[1].view(mc, batch_size, -1)  # mu ou time_pred
            if kind == "logspace":
                log_sigma = out[2].view(mc, batch_size, -1)
                preds = torch.clamp(torch.expm1(head), min=0)
                mean = preds.mean(0)
                epistemic = preds.std(0)
                aleatoric = torch.exp(log_sigma).mean(0) * mean  # delta method log→min
                std = torch.sqrt(epistemic**2 + aleatoric**2)
            else:  # minutes (bayésien) : seule l'incertitude MC
                preds = torch.clamp(head, min=0)
                mean = preds.mean(0)
                std = preds.std(0)
            return mean.cpu().numpy(), std.cpu().numpy()

        out = model(feats, mtimes, deltas, pmask)
        head = out[1]
        mean = torch.clamp(torch.expm1(head), min=0) if kind == "logspace" else torch.clamp(head, min=0)
        return mean.cpu().numpy(), None


def _confidence(pred_min: float, std_val: float | None) -> float:
    """Confiance d'une prédiction : Φ((30 − pred)/σ) si σ connu, sinon clip((30−pred)/30)."""
    if std_val is not None:
        return float(norm.cdf(30.0, pred_min, max(std_val, 1e-3)))
    return float(np.clip((30.0 - pred_min) / 30.0, 0.0, 1.0))


# ─── Inférence séquentielle : préfixe (one-pass) + queue (fenêtres glissantes) ──

def _emit_onepass(model, sessions: list[dict], device: torch.device, spec: dict,
                  n_mc: int) -> list[tuple]:
    """Préfixe de chaque session (≤ MAX_SEQ_LEN) en une passe → prédiction à chaque position.

    Émet les positions 2..min(n, MAX_SEQ_LEN)-1 (≥ 2 d'historique). Couvre intégralement
    les alertes ≤ 200 éclairs ; pour les plus longues, complété par `_emit_tail`.
    """
    mc = n_mc if spec["mc"] else 0
    per_batch = max(1, MC_CAP // n_mc) if spec["mc"] else POINT_BATCH

    prepared = []
    for s in sessions:
        times = np.asarray(s["times"], dtype=np.float64)
        win_len = min(len(times), MAX_SEQ_LEN)
        prepared.append({
            "features": np.asarray(s["features"], dtype=np.float32)[:win_len],
            "mtimes": (times[:win_len] - times[0]).astype(np.float32),
            "n_win": win_len,
            "orig_times": times,
            "start_time": s["start_time"],
            "airport": str(s["airport"]),
            "alert_id": str(s["alert_id"]),
        })
    prepared.sort(key=lambda p: p["n_win"])  # batches homogènes en longueur

    rows = []
    for batch in _chunks(prepared, per_batch):
        feats, mtimes, deltas, pmask = _pad_batch(batch, device)
        mean, std = _forward_batch(model, feats, mtimes, deltas, pmask, spec["kind"], mc)
        for b, p in enumerate(batch):
            for j in range(2, p["n_win"]):  # abs_idx == j (préfixe ancré à 0)
                pred_min = max(float(mean[b, j]), 0.0)
                pred_date = p["start_time"] + pd.Timedelta(minutes=float(p["orig_times"][j]))
                conf = _confidence(pred_min, None if std is None else float(std[b, j]))
                rows.append((p["airport"], p["alert_id"], pred_date,
                             pred_date + pd.Timedelta(minutes=pred_min), conf))
    return rows


def _emit_tail(model, sessions: list[dict], device: torch.device, spec: dict,
               n_mc: int) -> list[tuple]:
    """Queue des sessions > MAX_SEQ_LEN : une fenêtre glissante par événement t ≥ 200.

    Fenêtre = les MAX_SEQ_LEN éclairs finissant à t (réalignés à 0), prédiction lue à
    la dernière position. Garantit une prédiction à *tous* les événements des longues
    alertes, en gardant le modèle dans sa distribution d'entraînement (≤ 200 éclairs).
    """
    mc = n_mc if spec["mc"] else 0
    read_pos = MAX_SEQ_LEN - 1
    # Chaque fenêtre fait exactement MAX_SEQ_LEN événements → batch d'items par-batch.
    per_batch = max(1, MC_CAP // n_mc) if spec["mc"] else POINT_BATCH

    requests = []
    for s in sessions:
        times = np.asarray(s["times"], dtype=np.float64)
        features = np.asarray(s["features"], dtype=np.float32)
        n = len(times)
        for t in range(MAX_SEQ_LEN, n):
            lo = t - MAX_SEQ_LEN + 1
            requests.append({
                "features": features[lo : t + 1],
                "mtimes": (times[lo : t + 1] - times[lo]).astype(np.float32),
                "n_win": MAX_SEQ_LEN,
                "pred_date": s["start_time"] + pd.Timedelta(minutes=float(times[t])),
                "airport": str(s["airport"]),
                "alert_id": str(s["alert_id"]),
            })

    rows = []
    for batch in _chunks(requests, per_batch):
        feats, mtimes, deltas, pmask = _pad_batch(batch, device)
        mean, std = _forward_batch(model, feats, mtimes, deltas, pmask, spec["kind"], mc)
        for b, req in enumerate(batch):
            pred_min = max(float(mean[b, read_pos]), 0.0)
            conf = _confidence(pred_min, None if std is None else float(std[b, read_pos]))
            rows.append((req["airport"], req["alert_id"], req["pred_date"],
                         req["pred_date"] + pd.Timedelta(minutes=pred_min), conf))
    return rows


def _infer_sequential(model, sessions: list[dict], device: torch.device, spec: dict,
                      n_mc: int) -> pd.DataFrame:
    """Infère un modèle séquentiel sur des sessions → prédictions standard (tous événements)."""
    model.to(device)
    model.train() if spec["mc"] else model.eval()

    rows = _emit_onepass(model, sessions, device, spec, n_mc)
    long_sessions = [s for s in sessions if len(s["times"]) > MAX_SEQ_LEN]
    if long_sessions:
        rows += _emit_tail(model, long_sessions, device, spec, n_mc)
    return pd.DataFrame(rows, columns=PREDICTION_COLS)


def _build_model(spec: dict, models_dir: Path, device: torch.device):
    """Instancie un modèle depuis son registre et charge son checkpoint sur `device`."""
    import importlib

    module = importlib.import_module(spec["module"])
    model_cls = getattr(module, spec["cls"])
    model = model_cls(**spec["kwargs"])
    state = torch.load(str(models_dir / spec["file"]), map_location=device)
    model.load_state_dict(state)
    return model


def run_sequential_inference(
    sessions_by_kind: dict[str, list[dict]],
    specs: list[dict],
    models_dir: Path,
    devices: list[torch.device],
    n_mc: int = N_MC,
) -> dict[str, pd.DataFrame]:
    """Infère tous les modèles séquentiels présents, sessions shardées sur les devices.

    Chaque modèle voit ses sessions découpées en `len(devices)` shards entrelacés ; un
    thread par device traite ses shards. Les prédictions sont reconcaténées par modèle.
    """
    if not specs:
        return {}
    n_dev = len(devices)
    results: dict[str, list[pd.DataFrame | None]] = {s["name"]: [None] * n_dev for s in specs}

    def worker(d_idx: int) -> None:
        device = devices[d_idx]
        for spec in specs:
            sessions = sessions_by_kind[spec["prepare"]]
            shard = sessions[d_idx::n_dev]  # shard entrelacé (charge équilibrée)
            if not shard:
                results[spec["name"]][d_idx] = pd.DataFrame(columns=PREDICTION_COLS)
                continue
            model = _build_model(spec, models_dir, device)
            results[spec["name"]][d_idx] = _infer_sequential(model, shard, device, spec, n_mc)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_dev)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return {
        name: pd.concat([df for df in shards if df is not None], ignore_index=True)
        for name, shards in results.items()
    }


# ─── Inférence tabulaire (BNN, XGBoost) ───────────────────────────────────────

def _build_tabular_features(alerts: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit les features tabulaires de survie (même logique qu'à l'entraînement)."""
    import features as feat_module

    feat_df = feat_module.build_features(alerts)
    feat_df["icloud"] = feat_df["icloud"].astype(int)
    feat_df["airport"] = feat_df["airport"].astype(str)
    feat_df["airport_alert_id"] = feat_df["airport_alert_id"].astype(str)
    return feat_df


def _tabular_to_predictions(feat_df: pd.DataFrame, pred_min, std_min=None) -> pd.DataFrame:
    """Construit les prédictions standard depuis des prédictions ponctuelles (1/éclair)."""
    pred_min = np.clip(np.asarray(pred_min, dtype=float), 0.0, None)
    out = feat_df[["airport", "airport_alert_id", "date"]].copy()
    out = out.rename(columns={"date": "prediction_date"})
    out["predicted_date_end_alert"] = out["prediction_date"] + pd.to_timedelta(pred_min, unit="m")
    if std_min is not None:
        out["confidence"] = norm.cdf(30.0, loc=pred_min, scale=np.clip(std_min, 1e-3, None))
    else:
        out["confidence"] = np.clip((30.0 - pred_min) / 30.0, 0.0, 1.0)
    return out[PREDICTION_COLS]


def run_tabular_inference(
    alerts: pd.DataFrame, models_dir: Path, device: torch.device, n_samples: int = 100
) -> dict[str, pd.DataFrame]:
    """Infère les modèles tabulaires (BNN MC Dropout, XGBoost AFT) présents dans `models/`."""
    bnn_path = models_dir / "bnn_mc_dropout.pt"
    xgb_path = models_dir / "xgboost_aft.json"
    if not bnn_path.exists() and not xgb_path.exists():
        return {}

    feat_df = _build_tabular_features(alerts)
    predictions: dict[str, pd.DataFrame] = {}

    if bnn_path.exists():
        from bnn_model import BayesianMLP

        checkpoint = torch.load(str(bnn_path), map_location=device)
        cols = checkpoint["feature_cols"]
        mean_ = np.array(checkpoint["scaler_mean"])
        scale_ = np.array(checkpoint["scaler_scale"])
        model = BayesianMLP(input_dim=len(cols)).to(device)
        model.load_state_dict(checkpoint["model_state"])
        scaled = (feat_df[cols].astype(float).values - mean_) / scale_
        tensor = torch.FloatTensor(scaled).to(device)
        mean_pred, std_pred = model.predict_with_uncertainty(tensor, n_samples=n_samples)
        predictions["BNN MC Dropout"] = _tabular_to_predictions(feat_df, mean_pred, std_pred)

    if xgb_path.exists():
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(xgb_path))
        matrix = xgb.DMatrix(
            feat_df[TABULAR_FEATURE_COLS].values, feature_names=TABULAR_FEATURE_COLS
        )
        predictions["XGBoost AFT"] = _tabular_to_predictions(feat_df, booster.predict(matrix))

    return predictions


# ─── Baseline (ancre, rule-aware) ─────────────────────────────────────────────

def baseline_predictions(alerts: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Baseline 30 min : fin = dernier éclair pertinent + 30 min, une prédiction/alerte.

    Règle `"CG"` → dernier CG ; `"CG+IC"` → dernier éclair tous types. Par construction :
    gain 0 et risque 0 (ancre du protocole).
    """
    df = alerts.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["icloud"] = df["icloud"].astype(bool)
    df["airport"] = df["airport"].astype(str)
    df["airport_alert_id"] = df["airport_alert_id"].astype(str)
    if rule == "CG":
        df = df[~df["icloud"]]
    last = df.groupby(["airport", "airport_alert_id"])["date"].max().reset_index()
    last = last.rename(columns={"date": "prediction_date"})
    last["predicted_date_end_alert"] = last["prediction_date"] + pd.Timedelta(minutes=MAX_GAP_MIN)
    last["confidence"] = 1.0
    return last[PREDICTION_COLS]


# ─── Orchestration ────────────────────────────────────────────────────────────

def run_all_inference(
    alerts: pd.DataFrame,
    models_dir: Path | None = None,
    devices: list[torch.device] | None = None,
    n_mc: int = N_MC,
) -> dict[str, pd.DataFrame]:
    """Infère tous les modèles présents (séquentiels + tabulaires) → {nom: predictions_df}.

    La baseline n'est PAS incluse ici (elle dépend de la règle CG/CG+IC) : utiliser
    `baseline_predictions(alerts, rule)` au moment de l'évaluation.
    """
    models_dir = models_dir or MODELS_DIR
    devices = devices or detect_devices()
    print(f"[eval_inference] Devices : {[str(d) for d in devices]}")

    specs = [s for s in SEQ_REGISTRY if (models_dir / s["file"]).exists()]
    needed = {s["prepare"] for s in specs}

    sessions_by_kind: dict[str, list[dict]] = {}
    if "v2" in needed:
        import neural_hawkes_v2 as nh2

        print("[eval_inference] Préparation des sessions (12 features)…")
        sessions_by_kind["v2"] = nh2.prepare_sessions_v2(alerts)
    if "spatial" in needed:
        from spatial_features import prepare_sessions_spatial

        print("[eval_inference] Préparation des sessions spatiales (20 features)…")
        sessions_by_kind["spatial"] = prepare_sessions_spatial(alerts)

    predictions: dict[str, pd.DataFrame] = {}
    if specs:
        predictions.update(
            run_sequential_inference(sessions_by_kind, specs, models_dir, devices, n_mc)
        )
    else:
        print("[eval_inference] Aucun checkpoint séquentiel dans", models_dir)
    predictions.update(run_tabular_inference(alerts, models_dir, devices[0]))
    return predictions
