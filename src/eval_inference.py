"""eval_inference.py — Ré-inférence des modèles sauvegardés sur le jeu d'éval.

Charge les modèles entraînés (`models/*.json`, `models/*.pt`), refait l'inférence
sur le CSV d'éval officiel (2023+) et produit, pour chaque modèle, un DataFrame de
prédictions au format standard du jury attendu par `risk_gain_5km`.

GPU-aware : l'inférence utilise CUDA si disponible (env d'entraînement), sinon CPU
(env de développement). Les modèles absents de `models/` sont ignorés proprement.

Convention de prédiction :
  - Modèles tabulaires (XGBoost, BNN) : une prédiction par éclair observé.
  - Modèles séquentiels (Neural/Bayesian/MoE Hawkes) : une prédiction à plusieurs
    fractions d'avancement de chaque session (`FRACTIONS`), reconstruite en date
    absolue via `start_time + times[idx] + temps_restant_prédit`.
  - Confiance : pour une prédiction ponctuelle `clip((30 − pred) / 30, 0, 1)` ;
    pour un modèle à incertitude `P(restant < 30 | μ, σ) = Φ((30 − μ) / σ)`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# Les modules modèles s'importent mutuellement par nom court ; il faut src/ sur le path.
_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

import torch  # noqa: E402

import features as feat_module  # noqa: E402
from risk_gain_5km import PREDICTION_COLS  # noqa: E402

REPO_DIR = _SRC_DIR.parent
MODELS_DIR = REPO_DIR / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fractions d'avancement de session où l'on émet une prédiction (modèles séquentiels).
# Plus de points = plus de chances de trouver une prédiction précoce acceptée à theta donné.
FRACTIONS = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9]

# Nombre de passages Monte-Carlo pour les modèles à incertitude (MC Dropout / variationnel).
N_MC = 30

# Features tabulaires utilisées par XGBoost et le BNN (cf. evaluation.FEATURE_COLS).
TABULAR_FEATURE_COLS = [
    "time_since_start", "time_since_last_cg",
    "n_lightnings_15m", "n_lightnings_30m", "n_ic_15m", "n_ic_30m", "ic_ratio_15m",
    "mean_dist_15m", "mean_dist_30m", "dist_trend",
    "mean_abs_amp_15m", "max_abs_amp_15m",
    "dist", "amplitude", "maxis", "icloud",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]


# ─── Chargement / normalisation des données d'éval ────────────────────────────

def load_eval_alerts(csv_path: str | Path) -> pd.DataFrame:
    """Charge le CSV d'éval officiel et normalise son schéma.

    Le CSV jury utilise `alert_id` (et non `airport_alert_id`) et ne contient pas
    de `lightning_id`. On renomme, on filtre aux éclairs en alerte, on parse les
    dates en UTC et on ajoute un `lightning_id` séquentiel (requis par les features).

    Returns:
        DataFrame des éclairs en alerte, prêt pour l'inférence et l'évaluation.
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


# ─── Conversion prédictions ponctuelles → format standard ─────────────────────

def _tabular_to_predictions(
    base: pd.DataFrame, pred_min: np.ndarray, std_min: np.ndarray | None = None
) -> pd.DataFrame:
    """Construit le DataFrame standard depuis des prédictions ponctuelles (1/éclair).

    Args:
        base: doit contenir airport, airport_alert_id, date (UTC) — un éclair par ligne.
        pred_min: temps restant prédit (minutes) pour chaque éclair.
        std_min: écart-type prédit (minutes), optionnel → confiance calibrée.
    """
    pred_min = np.clip(np.asarray(pred_min, dtype=float), 0.0, None)
    out = base[["airport", "airport_alert_id", "date"]].copy()
    out = out.rename(columns={"date": "prediction_date"})
    out["predicted_date_end_alert"] = out["prediction_date"] + pd.to_timedelta(pred_min, unit="m")
    if std_min is not None:
        out["confidence"] = norm.cdf(30.0, loc=pred_min, scale=np.clip(std_min, 1e-3, None))
    else:
        out["confidence"] = np.clip((30.0 - pred_min) / 30.0, 0.0, 1.0)
    return out[PREDICTION_COLS]


# ─── Inférence tabulaire (XGBoost, BNN) ───────────────────────────────────────

def build_tabular_features(alerts: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit les features de survie sur l'éval (même logique que l'entraînement)."""
    feat_df = feat_module.build_features(alerts)
    feat_df["icloud"] = feat_df["icloud"].astype(int)
    return feat_df


def predict_xgboost(model_path: Path, feat_df: pd.DataFrame) -> pd.DataFrame:
    """Inférence XGBoost survival:AFT → prédictions standard (une par éclair)."""
    import xgboost as xgb

    model = xgb.Booster()
    model.load_model(str(model_path))
    matrix = xgb.DMatrix(
        feat_df[TABULAR_FEATURE_COLS].values, feature_names=TABULAR_FEATURE_COLS
    )
    pred_min = model.predict(matrix)
    return _tabular_to_predictions(feat_df, pred_min)


def predict_bnn(model_path: Path, feat_df: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
    """Inférence BNN MC Dropout → prédictions standard avec confiance calibrée."""
    from bnn_model import BayesianMLP

    checkpoint = torch.load(str(model_path), map_location=DEVICE)
    cols = checkpoint["feature_cols"]
    scaler_mean = np.array(checkpoint["scaler_mean"])
    scaler_scale = np.array(checkpoint["scaler_scale"])

    model = BayesianMLP(input_dim=len(cols)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    features_scaled = (feat_df[cols].astype(float).values - scaler_mean) / scaler_scale
    features_tensor = torch.FloatTensor(features_scaled).to(DEVICE)
    mean_pred, std_pred = model.predict_with_uncertainty(features_tensor, n_samples=n_samples)
    return _tabular_to_predictions(feat_df, mean_pred, std_pred)


# ─── Inférence séquentielle (Neural / Bayesian / MoE Hawkes) ──────────────────

def _sessions_to_predictions(
    sessions: list[dict], predict_fn, fractions: list[float], use_uncertainty: bool
) -> pd.DataFrame:
    """Émet une prédiction standard à chaque fraction d'avancement de chaque session.

    `predict_fn(features, times) -> (mean_min, std_min_or_None)` encapsule le modèle.
    La date de fin prédite est reconstruite : `start_time + times[idx] + mean_min`.
    """
    rows = []
    for session in sessions:
        times = session["times"]
        session_features = session["features"]
        start_time = session["start_time"]
        n_events = len(times)
        for frac in fractions:
            idx = int(n_events * frac)
            if idx < 2:
                continue
            mean_min, std_min = predict_fn(session_features[: idx + 1], times[: idx + 1])
            prediction_date = start_time + pd.Timedelta(minutes=float(times[idx]))
            predicted_end = prediction_date + pd.Timedelta(minutes=float(max(mean_min, 0.0)))
            if use_uncertainty and std_min is not None:
                confidence = float(norm.cdf(30.0, mean_min, max(float(std_min), 1e-3)))
            else:
                confidence = float(np.clip((30.0 - mean_min) / 30.0, 0.0, 1.0))
            rows.append(
                (session["airport"], session["alert_id"], prediction_date, predicted_end, confidence)
            )
    return pd.DataFrame(rows, columns=PREDICTION_COLS)


def _load_state_into(model, model_path: Path):
    """Charge un state_dict .pt sur le bon device et passe le modèle en eval."""
    state = torch.load(str(model_path), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# ─── Registre des modèles séquentiels (constructeur + recette d'inférence) ────
#
# Chaque entrée décrit comment reconstruire un modèle depuis son checkpoint et
# comment l'interroger. `prepare` vaut "v2" (12 features) ou "spatial" (20 features).
# `make_predict(trainer)` renvoie une fonction (features, times) -> (mean, std|None).

def _build_neural_registry():
    import neural_hawkes_v2 as nh2
    import neural_hawkes_v3 as nh3
    import bayesian_hawkes as bh
    import spatial_moe_model as moe

    def det(trainer):
        # Prédiction ponctuelle (pas d'incertitude).
        return lambda f, t: (trainer.predict_time_remaining(f, t), None)

    def unc_mc(trainer):
        # MC Dropout v3 / MoE (kwarg n_mc).
        return lambda f, t: trainer.predict_with_uncertainty(f, t, n_mc=N_MC)[:2]

    def unc_samples(trainer):
        # Bayésien (kwarg n_samples).
        return lambda f, t: trainer.predict_with_uncertainty(f, t, n_samples=N_MC)[:2]

    return [
        # Neural Hawkes V2 (régression directe, prédiction ponctuelle).
        {"name": "V2 GRU", "file": "neural_hawkes_gru_v2.pt", "prepare": "v2",
         "model_class": nh2.NeuralHawkesGRUv2,
         "kwargs": dict(input_dim=12, hidden_dim=64, dropout=0.1),
         "trainer_class": nh2.NeuralHawkesTrainer, "make_predict": det, "uncertainty": False},
        {"name": "V2 Transformer", "file": "neural_hawkes_transformer.pt", "prepare": "v2",
         "model_class": nh2.NeuralHawkesTransformer,
         "kwargs": dict(input_dim=12, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1),
         "trainer_class": nh2.NeuralHawkesTrainer, "make_predict": det, "uncertainty": False},
        {"name": "V2 Transformer-Large", "file": "neural_hawkes_transformer_large.pt", "prepare": "v2",
         "model_class": nh2.NeuralHawkesTransformer,
         "kwargs": dict(input_dim=12, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.15),
         "trainer_class": nh2.NeuralHawkesTrainer, "make_predict": det, "uncertainty": False},
        # Neural Hawkes V3 gaussien (incertitude MC Dropout).
        {"name": "V3 GRU gaussien", "file": "hawkes_gru_gaussian.pt", "prepare": "v2",
         "model_class": nh3.GaussianHawkesGRU,
         "kwargs": dict(input_dim=12, hidden_dim=64, dropout=0.15),
         "trainer_class": nh3.GaussianHawkesTrainer, "make_predict": unc_mc, "uncertainty": True},
        {"name": "V3 Transformer gaussien", "file": "hawkes_tf_gaussian.pt", "prepare": "v2",
         "model_class": nh3.GaussianHawkesTransformer,
         "kwargs": dict(input_dim=12, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.15),
         "trainer_class": nh3.GaussianHawkesTrainer, "make_predict": unc_mc, "uncertainty": True},
        {"name": "V3 Transformer-Large gaussien", "file": "hawkes_tf_large_gaussian.pt", "prepare": "v2",
         "model_class": nh3.GaussianHawkesTransformer,
         "kwargs": dict(input_dim=12, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.15),
         "trainer_class": nh3.GaussianHawkesTrainer, "make_predict": unc_mc, "uncertainty": True},
        # Hawkes bayésien (MC Dropout + variationnel).
        {"name": "Bayes MC Dropout", "file": "bayesian_hawkes_mc.pt", "prepare": "v2",
         "model_class": bh.MCDropoutHawkesTransformer,
         "kwargs": dict(input_dim=12, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.2),
         "trainer_class": bh.BayesianHawkesTrainer, "make_predict": unc_samples, "uncertainty": True},
        {"name": "Bayes Variationnel", "file": "bayesian_hawkes_var.pt", "prepare": "v2",
         "model_class": bh.VariationalHawkesTransformer,
         "kwargs": dict(input_dim=12, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1, prior_sigma=1.0),
         "trainer_class": bh.VariationalHawkesTrainer, "make_predict": unc_samples, "uncertainty": True},
        # Spatial MoE (20 features, incertitude MC Dropout).
        {"name": "Spatial MoE", "file": "spatial_moe.pt", "prepare": "spatial",
         "model_class": moe.SpatialMoETransformer,
         "kwargs": dict(input_dim=20, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.15),
         "trainer_class": moe.SpatialMoETrainer, "make_predict": unc_mc, "uncertainty": True},
    ]


def run_neural_inference(
    alerts: pd.DataFrame,
    fractions: list[float] | None = None,
    models_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Refait l'inférence de tous les modèles séquentiels présents dans `models/`.

    Prépare les sessions une seule fois par type de features (12 / 20), puis pour
    chaque checkpoint présent reconstruit le modèle, l'interroge et renvoie ses
    prédictions standard. Les modèles absents sont ignorés (avec un message).

    Returns:
        Dict {nom_modèle: predictions_df}. Vide si aucun checkpoint n'est présent.
    """
    fractions = fractions or FRACTIONS
    models_dir = models_dir or MODELS_DIR

    import neural_hawkes_v2 as nh2
    from spatial_features import prepare_sessions_spatial

    registry = _build_neural_registry()
    needed = {entry["prepare"] for entry in registry if (models_dir / entry["file"]).exists()}
    if not needed:
        print("Aucun modèle séquentiel trouvé dans", models_dir)
        return {}

    sessions_cache: dict[str, list[dict]] = {}
    if "v2" in needed:
        print("Préparation des sessions (12 features)...")
        sessions_cache["v2"] = nh2.prepare_sessions_v2(alerts)
    if "spatial" in needed:
        print("Préparation des sessions spatiales (20 features)...")
        sessions_cache["spatial"] = prepare_sessions_spatial(alerts)

    predictions: dict[str, pd.DataFrame] = {}
    for entry in registry:
        model_path = models_dir / entry["file"]
        if not model_path.exists():
            continue
        print(f"  Inférence {entry['name']} ({entry['file']}) sur {DEVICE}...")
        model = entry["model_class"](**entry["kwargs"])
        model = _load_state_into(model, model_path)
        trainer = entry["trainer_class"](model, device=DEVICE)
        predict_fn = entry["make_predict"](trainer)
        predictions[entry["name"]] = _sessions_to_predictions(
            sessions_cache[entry["prepare"]], predict_fn, fractions, entry["uncertainty"]
        )
    return predictions


def run_tabular_inference(
    alerts: pd.DataFrame, models_dir: Path | None = None
) -> dict[str, pd.DataFrame]:
    """Refait l'inférence des modèles tabulaires (XGBoost, BNN) présents dans `models/`."""
    models_dir = models_dir or MODELS_DIR
    xgb_path = models_dir / "xgboost_aft.json"
    bnn_path = models_dir / "bnn_mc_dropout.pt"
    if not xgb_path.exists() and not bnn_path.exists():
        print("Aucun modèle tabulaire trouvé dans", models_dir)
        return {}

    print("Construction des features tabulaires sur l'éval...")
    feat_df = build_tabular_features(alerts)

    predictions: dict[str, pd.DataFrame] = {}
    if xgb_path.exists():
        print(f"  Inférence XGBoost AFT ({xgb_path.name})...")
        predictions["XGBoost AFT"] = predict_xgboost(xgb_path, feat_df)
    if bnn_path.exists():
        print(f"  Inférence BNN MC Dropout ({bnn_path.name}) sur {DEVICE}...")
        predictions["BNN MC Dropout"] = predict_bnn(bnn_path, feat_df)
    return predictions


def baseline_predictions(alerts: pd.DataFrame) -> pd.DataFrame:
    """Baseline 30 min : fin d'alerte = dernier éclair + 30 min (règle métier actuelle).

    UNE seule prédiction par alerte (et non une par éclair) : le protocole retient
    la fin prédite la plus précoce par alerte, donc émettre `éclair + 30` à chaque
    éclair ferait retenir `premier_éclair + 30` au lieu de `dernier_éclair + 30`.
    Avec `dernier_éclair + 30`, la baseline est bien l'ancre attendue : gain 0 et
    risque 0 (aucun éclair après la fin, par construction).
    """
    last = (
        alerts.groupby(["airport", "airport_alert_id"])["date"].max().reset_index()
    )
    last = last.rename(columns={"date": "prediction_date"})
    last["predicted_date_end_alert"] = last["prediction_date"] + pd.Timedelta(minutes=30)
    last["confidence"] = 1.0
    return last[PREDICTION_COLS]


def run_all_inference(
    alerts: pd.DataFrame,
    fractions: list[float] | None = None,
    models_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Lance toute l'inférence disponible et renvoie {nom: predictions_df}."""
    predictions = {"Baseline 30 min": baseline_predictions(alerts)}
    predictions.update(run_tabular_inference(alerts, models_dir=models_dir))
    predictions.update(run_neural_inference(alerts, fractions=fractions, models_dir=models_dir))
    return predictions
