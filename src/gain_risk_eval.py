"""gain_risk_eval.py — Évaluation gain/risque indépendante de la fin d'alerte foudre.

Protocole (critères imposés, distincts du notebook jury) :

  - **Règle d'arrêt « first-passage »** : pour chaque alerte on parcourt les
    prédictions dans l'ordre chronologique (`prediction_date`) et on valide la
    *première* dont la confiance ≥ θ, puis on s'arrête. C'est `t_pred`. Aucune
    prédiction ≥ θ ⇒ alerte non couverte (repli baseline).

  - **Deux règles cohérentes par type** :
      * `"CG"`    : seuls les éclairs cloud-to-ground (`icloud == False`) comptent
                   (t_last, gain et risque ne portent que sur les CG).
      * `"CG+IC"` : tous les éclairs comptent, quel que soit le type.

  - **Gain jamais négatif** : fin effective `t_eff = min(t_pred, baseline_end)` où
    `baseline_end = t_last + 30 min`. Donc `gain = max(0, baseline_end − t_pred)` :
    une prédiction plus tardive que la baseline dégrade simplement vers la baseline.

  - **Risque** `R(d) = manqués / total` dans la zone `dist < d` (d = 5 puis 20 km).
    `total` = TOUS les éclairs pertinents de la zone sur **tout le jeu** (dénominateur
    global, comme le protocole jury) ; `manqués` = ceux survenant après `t_eff` dans
    une alerte couverte. Le risque utilise `t_eff` (donc le repli baseline ne manque
    rien).

  - **Calibration de θ** : `θ*(R) = argmax gain(R)` sous contrainte `R(5 km) <
    acceptable_risk` (2 %), balayée sur le jeu de calibration, puis figée et appliquée
    au holdout.

  - **Baseline** : ancre du protocole — gain 0 et risque 0 par construction.
"""

from __future__ import annotations

import pandas as pd

MAX_GAP_MIN = 30
ACCEPTABLE_RISK = 0.02
RULES = ("CG", "CG+IC")
ZONES_KM = (5.0, 20.0)

# Format standard des prédictions (identique au notebook jury).
PREDICTION_COLS = [
    "airport",
    "airport_alert_id",
    "prediction_date",
    "predicted_date_end_alert",
    "confidence",
]

# Grille de seuils par défaut : [0, 0.05, …, 0.95] (comme le notebook jury, n=20).
DEFAULT_THETAS = [i / 20 for i in range(20)]


def _normalize_alerts(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """Met les colonnes au bon type pour l'évaluation.

    Caster `airport`/`airport_alert_id` en `str` est crucial : un groupby sur une
    colonne catégorielle produit le produit cartésien des catégories (couples
    fantômes). `date` en UTC, `icloud` en booléen.
    """
    alerts = alerts_df.copy()
    alerts["date"] = pd.to_datetime(alerts["date"], utc=True)
    alerts["icloud"] = alerts["icloud"].astype(bool)
    alerts["airport"] = alerts["airport"].astype(str)
    alerts["airport_alert_id"] = alerts["airport_alert_id"].astype(str)
    return alerts


def _normalize_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise et trie les prédictions par date d'émission (requis pour first-passage)."""
    preds = predictions_df.copy()
    preds["prediction_date"] = pd.to_datetime(preds["prediction_date"], utc=True)
    preds["predicted_date_end_alert"] = pd.to_datetime(
        preds["predicted_date_end_alert"], utc=True
    )
    preds["airport"] = preds["airport"].astype(str)
    preds["airport_alert_id"] = preds["airport_alert_id"].astype(str)
    return preds.sort_values("prediction_date")


def _relevant_mask(alerts: pd.DataFrame, rule: str) -> pd.Series:
    """Masque des éclairs pertinents pour la règle (`CG` → cloud-to-ground seuls)."""
    if rule == "CG":
        return ~alerts["icloud"]
    if rule == "CG+IC":
        return pd.Series(True, index=alerts.index)
    raise ValueError(f"Règle inconnue : {rule!r} (attendu : {RULES})")


def first_passage(predictions_df: pd.DataFrame, theta: float) -> pd.Series:
    """Sélectionne, par alerte, la fin prédite de la 1ʳᵉ prédiction franchissant θ.

    Args:
        predictions_df: prédictions au format `PREDICTION_COLS`, supposées triées par
            `prediction_date` croissant (`_normalize_predictions`).
        theta: seuil de confiance.

    Returns:
        Série indexée par `(airport, airport_alert_id)` donnant `t_pred`
        (`predicted_date_end_alert` de la première prédiction de confiance ≥ θ).
        Les alertes sans aucune prédiction ≥ θ sont absentes (non couvertes).
    """
    accepted = predictions_df[predictions_df["confidence"] >= theta]
    if accepted.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    # Déjà triées par prediction_date → la première de chaque groupe est la plus précoce.
    return accepted.groupby(["airport", "airport_alert_id"])[
        "predicted_date_end_alert"
    ].first()


def evaluate(
    predictions_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    theta: float,
    rule: str,
    min_dist_km: float,
    max_gap_min: int = MAX_GAP_MIN,
) -> dict:
    """Calcule gain (heures) et risque pour un θ, une règle et une zone donnés.

    Args:
        predictions_df: prédictions standard (non encore normalisées).
        alerts_df: éclairs bruts en alerte (airport, airport_alert_id, date, dist, icloud).
        theta: seuil de confiance.
        rule: `"CG"` ou `"CG+IC"`.
        min_dist_km: rayon de la zone dangereuse (5 ou 20 km).
        max_gap_min: durée de la règle baseline (30 min).

    Returns:
        Dict {theta, rule, min_dist_km, gain_h, risk, missed, total, n_covered,
        n_alerts}. `total` est le dénominateur global (tous les éclairs pertinents
        de la zone, tout le jeu).
    """
    alerts = _normalize_alerts(alerts_df)
    preds = _normalize_predictions(predictions_df)

    relevant = alerts[_relevant_mask(alerts, rule)]
    grouped = relevant.groupby(["airport", "airport_alert_id"])

    # Dénominateur global : tous les éclairs pertinents de la zone sur tout le jeu.
    total = int((relevant["dist"] < min_dist_km).sum())

    committed = first_passage(preds, theta)

    gain_seconds = 0.0
    missed = 0
    for (airport, alert_id), t_pred in committed.items():
        if (airport, alert_id) not in grouped.groups:
            # Alerte présente dans les prédictions mais sans éclair pertinent
            # (p.ex. règle CG sur une alerte 100 % IC) → rien à gagner ni manquer.
            continue
        group = grouped.get_group((airport, alert_id))
        baseline_end = group["date"].max() + pd.Timedelta(minutes=max_gap_min)
        # Plafond : on ne reste jamais ouvert plus longtemps que la baseline.
        t_eff = min(t_pred, baseline_end)
        gain_seconds += (baseline_end - t_eff).total_seconds()

        danger = group[group["dist"] < min_dist_km]
        missed += int((danger["date"] > t_eff).sum())

    return {
        "theta": theta,
        "rule": rule,
        "min_dist_km": min_dist_km,
        "gain_h": gain_seconds / 3600.0,
        "risk": missed / total if total else 0.0,
        "missed": missed,
        "total": total,
        "n_covered": int(committed.shape[0]),
        "n_alerts": int(relevant.groupby(["airport", "airport_alert_id"]).ngroups),
    }


def sweep(
    predictions_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    rule: str,
    min_dist_km: float,
    thetas: list[float] | None = None,
) -> pd.DataFrame:
    """Balaye une grille de θ et renvoie une ligne de métriques par θ."""
    thetas = thetas if thetas is not None else DEFAULT_THETAS
    rows = [
        evaluate(predictions_df, alerts_df, theta, rule, min_dist_km) for theta in thetas
    ]
    return pd.DataFrame(rows)


def calibrate_theta(
    predictions_df: pd.DataFrame,
    calib_alerts_df: pd.DataFrame,
    rule: str,
    acceptable_risk: float = ACCEPTABLE_RISK,
    thetas: list[float] | None = None,
) -> dict:
    """Choisit θ* maximisant le gain sous contrainte de risque 5 km < `acceptable_risk`.

    La contrainte porte toujours sur la zone **5 km** de la règle évaluée (zone
    critique de sécurité).

    Returns:
        Dict {theta_star, feasible, gain_h, risk_5km, sweep_df}. Si aucun θ n'est
        faisable, `feasible=False` et `theta_star` = θ de risque 5 km minimal.
    """
    sweep_df = sweep(predictions_df, calib_alerts_df, rule, min_dist_km=5.0, thetas=thetas)
    feasible = sweep_df[sweep_df["risk"] < acceptable_risk]
    if not feasible.empty:
        best = feasible.loc[feasible["gain_h"].idxmax()]
        is_feasible = True
    else:
        # Aucun θ ne respecte 2 % : on retient le moins risqué (à signaler).
        best = sweep_df.loc[sweep_df["risk"].idxmin()]
        is_feasible = False
    return {
        "theta_star": float(best["theta"]),
        "feasible": is_feasible,
        "gain_h": float(best["gain_h"]),
        "risk_5km": float(best["risk"]),
        "sweep_df": sweep_df,
    }


def apply_holdout(
    predictions_df: pd.DataFrame,
    holdout_alerts_df: pd.DataFrame,
    theta_star: float,
    rule: str,
    zones_km: tuple[float, ...] = ZONES_KM,
) -> dict:
    """Applique un θ* figé au holdout et renvoie gain + risque pour chaque zone.

    Returns:
        Dict {theta, rule, gain_h, n_covered, n_alerts, risk_5km, missed_5km,
        total_5km, risk_20km, missed_20km, total_20km, …} (une triplette par zone).
    """
    out: dict = {"theta": theta_star, "rule": rule}
    for dist_km in zones_km:
        res = evaluate(predictions_df, holdout_alerts_df, theta_star, rule, dist_km)
        suffix = f"{int(dist_km)}km"
        out[f"risk_{suffix}"] = res["risk"]
        out[f"missed_{suffix}"] = res["missed"]
        out[f"total_{suffix}"] = res["total"]
        # gain et couverture sont indépendants de la zone : on les pose une fois.
        out["gain_h"] = res["gain_h"]
        out["n_covered"] = res["n_covered"]
        out["n_alerts"] = res["n_alerts"]
    return out
