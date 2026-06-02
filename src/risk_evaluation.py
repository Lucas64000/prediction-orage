"""
risk_evaluation.py - Évaluation gain/risque selon le protocole du hackathon.

Pour chaque modèle, à theta donné :
  - gain_h : heures gagnées vs baseline 30 min
  - risk   : éclairs < 3 km ratés / total éclairs < 3 km (cible < 2 %)

Toutes les prédictions sont exprimées comme un DataFrame standardisé avec les
colonnes : airport, airport_alert_id, prediction_date, predicted_date_end_alert,
confidence.

Schéma de la confiance pour les prédictions ponctuelles :
    confidence = clip((30 - pred_min) / 30, 0, 1)
  → pred = 0 min → confiance = 1.0
  → pred = 18 min → confiance = 0.4  (seuil theta recommandé)
  → pred ≥ 30 min → confiance = 0.0  (pas mieux que la baseline)

Pour les modèles gaussiens avec incertitude (sigma), on peut utiliser en option
P(pred < 30 | μ, σ) = Φ((30 − μ) / σ) comme confiance calibrée.
"""

import numpy as np
import pandas as pd

MAX_GAP_MIN = 30
MIN_DIST_KM = 3.0


def predictions_from_tabular(val_df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """
    Construit le DataFrame de prédictions depuis un modèle tabulaire (XGBoost, BNN…).

    Args:
        val_df  : DataFrame de validation avec colonnes airport, airport_alert_id,
                  date (datetime), et pred_col (minutes restantes prédites).
        pred_col: Nom de la colonne de prédiction dans val_df.

    Returns:
        DataFrame au format standard de prédictions.
    """
    df = val_df[["airport", "airport_alert_id", "date", pred_col]].copy()
    df["prediction_date"] = pd.to_datetime(df["date"], utc=True)
    df["predicted_date_end_alert"] = df["prediction_date"] + pd.to_timedelta(
        df[pred_col].clip(lower=0), unit="m"
    )
    df["confidence"] = ((MAX_GAP_MIN - df[pred_col]) / MAX_GAP_MIN).clip(0, 1)
    return df[
        ["airport", "airport_alert_id", "prediction_date", "predicted_date_end_alert", "confidence"]
    ]


def predictions_from_errors_df(
    errors_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    use_uncertainty: bool = False,
) -> pd.DataFrame:
    """
    Construit le DataFrame de prédictions depuis un errors_df de modèle neural.

    Les modèles neuraux évaluent à des fractions de session (30 %, 50 %, 70 %, 90 %).
    On reconstruit la datetime absolue de fin prédite via :
        predicted_end = actual_last_cg + (pred − true)
    ce qui est exact car : event_t + pred = (last_cg − true) + pred = last_cg + (pred − true).

    Args:
        errors_df       : DataFrame avec colonnes airport, airport_alert_id, true, pred
                          (et optionnellement uncertainty pour les modèles gaussiens).
        alerts_df       : DataFrame brut des alertes (airport, airport_alert_id, date, dist).
        use_uncertainty : Si True et que la colonne uncertainty est présente, calcule
                          la confiance via P(pred < 30 | μ, σ) = Φ((30 − μ) / σ).

    Returns:
        DataFrame au format standard de prédictions.
    """
    required = {"airport", "airport_alert_id", "true", "pred"}
    missing = required - set(errors_df.columns)
    if missing:
        raise ValueError(f"errors_df manque les colonnes : {missing}")

    # Datetime du dernier éclair par session (= fin réelle de l'alerte)
    last_cg = (
        alerts_df.groupby(["airport", "airport_alert_id"])["date"]
        .max()
        .reset_index()
        .rename(columns={"date": "last_cg_dt"})
    )
    last_cg["last_cg_dt"] = pd.to_datetime(last_cg["last_cg_dt"], utc=True)

    df = errors_df.merge(last_cg, on=["airport", "airport_alert_id"], how="left")
    df = df.dropna(subset=["last_cg_dt"])

    # Décalage entre prédiction et réalité
    offset_min = df["pred"].astype(float) - df["true"].astype(float)
    df["predicted_date_end_alert"] = df["last_cg_dt"] + pd.to_timedelta(offset_min, unit="m")
    df["prediction_date"] = df["last_cg_dt"] - pd.to_timedelta(df["true"].astype(float), unit="m")

    if use_uncertainty and "uncertainty" in df.columns:
        # Confiance calibrée : P(temps restant < 30 min | μ, σ) via loi normale
        from scipy.stats import norm
        df["confidence"] = norm.cdf(
            MAX_GAP_MIN,
            loc=df["pred"].astype(float),
            scale=df["uncertainty"].astype(float).clip(lower=1e-3),
        )
    else:
        df["confidence"] = ((MAX_GAP_MIN - df["pred"].astype(float)) / MAX_GAP_MIN).clip(0, 1)

    return df[
        ["airport", "airport_alert_id", "prediction_date", "predicted_date_end_alert", "confidence"]
    ]


def evaluate_at_theta(
    predictions_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    theta: float = 0.4,
    max_gap_min: int = MAX_GAP_MIN,
    min_dist_km: float = MIN_DIST_KM,
) -> dict:
    """
    Calcule gain et risque pour un jeu de prédictions au seuil theta.

    Args:
        predictions_df : DataFrame standard (airport, airport_alert_id,
                         predicted_date_end_alert, confidence).
        alerts_df      : DataFrame brut des alertes (airport, airport_alert_id, date, dist).
        theta          : Seuil de confiance minimum pour accepter une prédiction.
        max_gap_min    : Durée de la règle baseline en minutes (défaut : 30).
        min_dist_km    : Distance dangereuse en km (défaut : 3).

    Returns:
        Dict avec gain_h, gain_min, risk, missed, tot_lightnings_3km, covered_alerts.
    """
    covered_ids = predictions_df["airport_alert_id"].unique()
    test_alerts = alerts_df[alerts_df["airport_alert_id"].isin(covered_ids)]
    tot_3km = int((test_alerts["dist"] < min_dist_km).sum())

    accepted = predictions_df[predictions_df["confidence"] >= theta]
    if len(accepted) == 0:
        return {
            "gain_h": 0.0, "gain_min": 0.0, "risk": float("inf"),
            "missed": 0, "tot_lightnings_3km": tot_3km, "covered_alerts": 0,
        }

    # Prédiction la plus optimiste (la plus précoce) par alerte
    best = (
        accepted.groupby(["airport", "airport_alert_id"])["predicted_date_end_alert"]
        .min()
        .reset_index()
    )

    alerts_grouped = alerts_df.groupby(["airport", "airport_alert_id"])
    gain_s = 0.0
    missed = 0

    for _, row in best.iterrows():
        key = (row["airport"], row["airport_alert_id"])
        if key not in alerts_grouped.groups:
            continue
        group = alerts_grouped.get_group(key)
        end_baseline = (
            pd.to_datetime(group["date"], utc=True).max()
            + pd.Timedelta(minutes=max_gap_min)
        )
        pred_end = row["predicted_date_end_alert"]
        gain_s += (end_baseline - pred_end).total_seconds()

        dangerous = group[group["dist"] < min_dist_km]
        missed += int((pd.to_datetime(dangerous["date"], utc=True) > pred_end).sum())

    risk = missed / tot_3km if tot_3km > 0 else 0.0

    return {
        "gain_h": gain_s / 3600,
        "gain_min": gain_s / 60,
        "risk": risk,
        "missed": missed,
        "tot_lightnings_3km": tot_3km,
        "covered_alerts": len(best),
    }


def print_risk_table(results: dict, theta: float, acceptable_risk: float = 0.02) -> None:
    """Affiche un tableau comparatif gain / risque pour tous les modèles."""
    print(f"\n{'='*72}")
    print(f"  ÉVALUATION RISQUE / GAIN  (θ={theta}, risque acceptable < {acceptable_risk*100:.0f} %)")
    print(f"{'='*72}")
    print(f"  {'Modèle':35s} | {'Gain (h)':>8s} | {'Risque':>7s} | {'Ratés':>6s} | {'Alertes':>7s}")
    print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")
    for name, r in results.items():
        risk_str = f"{r['risk']*100:.2f} %"
        ok = "✓" if r["risk"] <= acceptable_risk else "✗"
        print(
            f"  {name:35s} | {r['gain_h']:8.1f} | {ok} {risk_str:>5s} | "
            f"{r['missed']:6d} | {r['covered_alerts']:7d}"
        )
    print(f"{'='*72}")
