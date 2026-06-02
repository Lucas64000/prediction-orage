"""risk_gain_5km.py — Protocole d'évaluation gain / risque du hackathon, étendu.

Reprend fidèlement le protocole officiel (`Evaluation_databattle_meteorage.ipynb`)
mais généralise la distance dangereuse (5 km par défaut au lieu de 3 km) et
décompose le risque par type d'éclair (CG = cloud-to-ground, IC = intra-cloud).

Définitions (toutes les dates sont comparées en UTC) :
  - Gain : pour chaque alerte couverte, `(t_last + 30 min) − t_pred` sommé sur
    toutes les alertes, où `t_last` est le dernier éclair de l'alerte (CG ou IC)
    et `t_pred` la fin d'alerte prédite. C'est le temps gagné vs la baseline 30 min.
  - Risque : `R = M / N` où, dans la zone `dist < min_dist` :
      N = nombre total d'éclairs,
      M = nombre d'éclairs « manqués », c.-à-d. survenus APRÈS la fin prédite
          (`t_pred < t_i`) dans l'alerte concernée.
    On calcule R global (CG+IC) et sa décomposition R_cg, R_ic.

Sélection de prédiction : à un seuil de confiance `theta`, on retient par alerte
la prédiction de fin la plus précoce parmi celles de confiance ≥ theta
(`t_pred = min_i t_pred_i, s_i ≥ theta`).

Le tot d'éclairs (dénominateur N) est restreint aux alertes effectivement
couvertes par le modèle, pour comparer les modèles sur leur propre périmètre.
"""

from __future__ import annotations

import pandas as pd

MAX_GAP_MIN = 30
MIN_DIST_KM = 5.0

# Colonnes du DataFrame de prédictions au format standard du jury.
PREDICTION_COLS = [
    "airport",
    "airport_alert_id",
    "prediction_date",
    "predicted_date_end_alert",
    "confidence",
]


def default_thetas(n_samples: int = 20) -> list[float]:
    """Grille de seuils de confiance régulière sur [0, 1[ (comme le notebook jury)."""
    return [i / n_samples for i in range(n_samples)]


def _prepare_alerts(alerts_df: pd.DataFrame, min_dist_km: float) -> pd.DataFrame:
    """Normalise les colonnes nécessaires et restreint à la zone dangereuse utile.

    On garde toutes les lignes (pour calculer `t_last` = dernier éclair par alerte)
    mais on s'assure que `date` est en UTC et `icloud` booléen.
    """
    alerts = alerts_df.copy()
    alerts["date"] = pd.to_datetime(alerts["date"], utc=True)
    alerts["icloud"] = alerts["icloud"].astype(bool)
    return alerts


def evaluate_theta_sweep(
    predictions_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    thetas: list[float] | None = None,
    min_dist_km: float = MIN_DIST_KM,
    max_gap_min: int = MAX_GAP_MIN,
) -> tuple[pd.DataFrame, dict]:
    """Balaye les seuils `theta` et calcule gain + risque (CG/IC) pour chacun.

    Args:
        predictions_df: prédictions au format standard (PREDICTION_COLS).
        alerts_df: éclairs bruts en alerte (airport, airport_alert_id, date, dist, icloud).
        thetas: grille de seuils. Défaut : `default_thetas()`.
        min_dist_km: distance en-dessous de laquelle un éclair est dangereux (5 km).
        max_gap_min: durée de la règle baseline (30 min).

    Returns:
        (sweep_df, totals) où `sweep_df` a une ligne par theta avec les colonnes
        theta, gain_h, missed_total/cg/ic, risk_total/cg/ic, n_alerts_covered ;
        `totals` donne les dénominateurs tot_total/tot_cg/tot_ic et la couverture.
    """
    if thetas is None:
        thetas = default_thetas()

    preds = predictions_df.copy()
    preds["predicted_date_end_alert"] = pd.to_datetime(
        preds["predicted_date_end_alert"], utc=True
    )
    alerts = _prepare_alerts(alerts_df, min_dist_km)

    # Dénominateur N : éclairs en zone, restreints aux alertes que le modèle couvre.
    covered_ids = set(preds["airport_alert_id"].unique())
    covered_alerts = alerts[alerts["airport_alert_id"].isin(covered_ids)]
    near = covered_alerts["dist"] < min_dist_km
    tot_total = int(near.sum())
    tot_cg = int((near & ~covered_alerts["icloud"]).sum())
    tot_ic = int((near & covered_alerts["icloud"]).sum())

    grouped = alerts.groupby(["airport", "airport_alert_id"])

    rows = []
    for theta in thetas:
        accepted = preds[preds["confidence"] >= theta]
        if len(accepted) == 0:
            rows.append(
                {
                    "theta": theta, "gain_h": 0.0,
                    "missed_total": 0, "missed_cg": 0, "missed_ic": 0,
                    "risk_total": 0.0, "risk_cg": 0.0, "risk_ic": 0.0,
                    "n_alerts_covered": 0,
                }
            )
            continue

        # Prédiction de fin la plus précoce par alerte parmi celles ≥ theta.
        best_end = (
            accepted.groupby(["airport", "airport_alert_id"])["predicted_date_end_alert"]
            .min()
        )

        gain_s = 0.0
        missed_total = missed_cg = missed_ic = 0
        for (airport, alert_id), pred_end in best_end.items():
            group = grouped.get_group((airport, alert_id))
            baseline_end = group["date"].max() + pd.Timedelta(minutes=max_gap_min)
            gain_s += (baseline_end - pred_end).total_seconds()

            danger_zone = group[group["dist"] < min_dist_km]
            late = danger_zone["date"] > pred_end
            missed_total += int(late.sum())
            missed_cg += int((late & ~danger_zone["icloud"]).sum())
            missed_ic += int((late & danger_zone["icloud"]).sum())

        rows.append(
            {
                "theta": theta,
                "gain_h": gain_s / 3600,
                "missed_total": missed_total,
                "missed_cg": missed_cg,
                "missed_ic": missed_ic,
                "risk_total": missed_total / tot_total if tot_total else 0.0,
                "risk_cg": missed_cg / tot_cg if tot_cg else 0.0,
                "risk_ic": missed_ic / tot_ic if tot_ic else 0.0,
                "n_alerts_covered": int(best_end.shape[0]),
            }
        )

    totals = {
        "tot_total": tot_total,
        "tot_cg": tot_cg,
        "tot_ic": tot_ic,
        "n_alerts_covered_max": len(covered_ids),
        "min_dist_km": min_dist_km,
    }
    return pd.DataFrame(rows), totals


def select_best_theta(
    sweep_df: pd.DataFrame,
    acceptable_risk: float = 0.02,
    risk_col: str = "risk_total",
) -> dict | None:
    """Sélectionne le theta de gain maximal respectant `risk_col < acceptable_risk`.

    Returns:
        La ligne du sweep (en dict) du meilleur theta, ou None si aucun seuil ne
        respecte la contrainte de risque.
    """
    feasible = sweep_df[sweep_df[risk_col] < acceptable_risk]
    if feasible.empty:
        return None
    best = feasible.loc[feasible["gain_h"].idxmax()]
    return best.to_dict()
