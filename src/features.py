import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features séquentielles et les cibles de temps restant pour l'analyse de survie.
    On s'attend à recevoir le DataFrame des alertes uniquement.
    """
    # Trier chronologiquement par session
    df = df.sort_values(["airport_alert_id", "date"]).copy()

    features_list = []

    groups = df.groupby(["airport", "airport_alert_id"])
    for (airport, alert_id), group in tqdm(
        groups, desc="Feature engineering par session"
    ):
        start_time = group["date"].iloc[0]

        # Le dernier CG definit la fin (règle métier)
        cg_only = group[~group["icloud"]].sort_values("date")
        if len(cg_only) == 0:
            continue

        last_cg_time = cg_only["date"].iloc[-1]

        # On ignore les éclairs postérieurs à la fin de l'orage officielle (dernier CG)
        group = group[group["date"] <= last_cg_time].copy()
        if len(group) == 0:
            continue

        # --- CIBLE (TARGET) ---
        group["time_to_end"] = (last_cg_time - group["date"]).dt.total_seconds() / 60.0

        # Feature de base
        group["time_since_start"] = (
            group["date"] - start_time
        ).dt.total_seconds() / 60.0

        # Temps depuis le dernier CG
        # On fait un merge asof sur les CG précédents
        group_cg = cg_only[cg_only["date"] <= last_cg_time].copy()

        group = pd.merge_asof(
            group,
            group_cg[["date", "lightning_id"]].rename(
                columns={"date": "last_cg_date", "lightning_id": "last_cg_id"}
            ),
            left_on="date",
            right_on="last_cg_date",
            direction="backward",
        )
        group["time_since_last_cg"] = (
            group["date"] - group["last_cg_date"]
        ).dt.total_seconds() / 60.0
        # Quand c'est le 1er CG, time_since_last_cg = 0
        group["time_since_last_cg"] = group["time_since_last_cg"].fillna(0)

        # --- ROLLING WINDOWS (15 min) ---
        group = group.set_index("date")

        r15 = group.rolling("15min", closed="right")
        r30 = group.rolling("30min", closed="right")

        # Dénombrement
        group["n_lightnings_15m"] = r15["lightning_id"].count()
        group["n_lightnings_30m"] = r30["lightning_id"].count()

        group["n_ic_15m"] = r15["icloud"].sum()
        group["n_ic_30m"] = r30["icloud"].sum()

        group["ic_ratio_15m"] = group["n_ic_15m"] / group["n_lightnings_15m"].replace(
            0, 1
        )

        # Distribution physique
        group["mean_dist_15m"] = r15["dist"].mean()
        group["mean_dist_30m"] = r30["dist"].mean()

        # Vitesse d'approche : si positif, l'orage recule (distance augmente). Si négatif, l'orage s'approche.
        group["dist_trend"] = group["mean_dist_15m"] - group["mean_dist_30m"]

        # Intensité / Amplitude (en valeur absolue)
        group["abs_amp"] = group["amplitude"].abs()
        group["mean_abs_amp_15m"] = group.rolling("15min", closed="right")[
            "abs_amp"
        ].mean()
        group["max_abs_amp_15m"] = group.rolling("15min", closed="right")[
            "abs_amp"
        ].max()

        group = group.reset_index()
        features_list.append(group)

    res = pd.concat(features_list, ignore_index=True)

    # Encodage cyclique classique (mois, heure)
    res["month"] = res["date"].dt.month
    res["hour"] = res["date"].dt.hour

    res["hour_sin"] = np.sin(2 * np.pi * res["hour"] / 24)
    res["hour_cos"] = np.cos(2 * np.pi * res["hour"] / 24)
    res["month_sin"] = np.sin(2 * np.pi * res["month"] / 12)
    res["month_cos"] = np.cos(2 * np.pi * res["month"] / 12)

    # Variables catégorielles (aéroport)
    res["airport"] = res["airport"].astype("category")

    # Nettoyage
    # fillna uniquement sur les colonnes numériques
    num_cols = res.select_dtypes(exclude="category").columns
    res[num_cols] = res[num_cols].fillna(0)

    # is_last_lightning_cloud_ground est une colonne source (object avec bool+NaN)
    # qui bloque la sérialisation parquet. On la supprime ; is_last_cg (booléen
    # nullable) en est déjà la version parsée.
    res = res.drop(columns=["is_last_lightning_cloud_ground"], errors="ignore")
    if "is_last_cg" in res.columns:
        res["is_last_cg"] = res["is_last_cg"].astype("boolean")

    return res


if __name__ == "__main__":
    from data_loader import load_raw, load_alerts

    print("Loading data...")
    df = load_raw()
    alerts = load_alerts(df)

    print("Building features...")
    feat_df = build_features(alerts)

    print("Features shape:", feat_df.shape)

    data_dir = Path(__file__).parent.parent / "data"
    out_path = data_dir / "features_survival.parquet"
    feat_df.to_parquet(out_path, index=False)
    print(f"Features saved to {out_path}")
