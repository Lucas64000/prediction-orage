"""Construit les CSV d'alertes pour les modèles à partir des éclairs bruts.

Lit `raw_data/*.csv` (un fichier par aéroport), segmente les éclairs en alertes
selon la règle « tout éclair en zone (CG ou IC) déclenche / prolonge l'alerte »,
puis écrit les CSV global, train et eval dans `data/`.

Schéma de sortie (compatible `src/data_loader.py`) :
    lightning_id, airport, airport_alert_id, date,
    is_last_lightning_cloud_ground, dist, lon, lat, azimuth,
    amplitude, maxis, icloud
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_RAW_DATA_DIR: Path = PROJECT_ROOT.parent / "raw_data"
DEFAULT_DATA_DIR: Path = PROJECT_ROOT / "data"

# Seuils métier de la règle d'alerte.
MAX_DISTANCE_KM: float = 20.0  # Rayon de la zone d'alerte autour de l'aéroport.
MAX_GAP_MINUTES: float = 30.0  # Sans éclair pendant ce délai, l'alerte est levée.

# Bornes du split temporel train/eval (saisons d'orage entières, pas d'aléatoire).
TRAIN_END_YEAR: int = 2022
EVAL_START_YEAR: int = 2023

OUTPUT_COLUMNS: list[str] = [
    "lightning_id",
    "airport",
    "airport_alert_id",
    "date",
    "is_last_lightning_cloud_ground",
    "dist",
    "lon",
    "lat",
    "azimuth",
    "amplitude",
    "maxis",
    "icloud",
]


def load_raw_airports(raw_dir: Path) -> pd.DataFrame:
    """Concatène tous les CSV bruts d'aéroports avec une colonne `airport` taguée."""
    frames = []
    for csv_file in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(csv_file)
        df["airport"] = csv_file.stem
        df["date"] = pd.to_datetime(df["date"], utc=True)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def segment_alerts(df_lightnings: pd.DataFrame) -> pd.DataFrame:
    """Segmente les éclairs en alertes, indépendamment pour chaque aéroport.

    Tout éclair en zone (`dist <= MAX_DISTANCE_KM`) reçoit un `airport_alert_id` ;
    les éclairs hors zone reçoivent NaN. Deux éclairs en zone d'un même aéroport
    appartiennent à la même alerte si leur écart est `<= MAX_GAP_MINUTES`.

    Colonnes ajoutées :
    - `lightning_id` : index global après tri (airport, date).
    - `airport_alert_id` : `"{airport}_{n}"`, NaN hors zone.
    - `is_last_lightning_cloud_ground` : True pour le dernier CG de chaque alerte,
      False pour les autres CG en alerte, NaN sinon (IC ou hors alerte).

    Args:
        df_lightnings: DataFrame d'éclairs avec au moins `airport`, `date`
            (datetime UTC), `dist` (km) et `icloud` (bool).

    Returns:
        DataFrame trié par `(airport, date)` et restreint à `OUTPUT_COLUMNS`.
    """
    df = df_lightnings.sort_values(["airport", "date"]).reset_index(drop=True)
    df["lightning_id"] = df.index

    # Éclairs en zone d'alerte : eux seuls déclenchent et maintiennent une alerte.
    in_alert_mask = df["dist"] <= MAX_DISTANCE_KM
    df_in = df.loc[in_alert_mask].copy()

    # Délai vers l'éclair en zone précédent du même aéroport (NaN pour le 1er).
    time_since_prev = (
        df_in.groupby("airport")["date"].diff().dt.total_seconds() / 60
    )
    # Nouvelle alerte : 1er éclair en zone d'un aéroport, ou gap strict > MAX_GAP_MINUTES.
    is_new_alert = time_since_prev.isna() | (time_since_prev > MAX_GAP_MINUTES)
    alert_num = is_new_alert.groupby(df_in["airport"]).cumsum()
    df_in["airport_alert_id"] = df_in["airport"] + "_" + alert_num.astype(str)

    df = df.join(df_in[["airport_alert_id"]])

    # is_last_lightning_cloud_ground : True pour le dernier CG de chaque alerte,
    # False pour les autres CG en alerte, NaN partout ailleurs (IC ou hors alerte).
    cg_in_alert_mask = df["airport_alert_id"].notna() & ~df["icloud"]
    df["is_last_lightning_cloud_ground"] = pd.NA
    if cg_in_alert_mask.any():
        cg_in_alert = df.loc[cg_in_alert_mask]
        is_last_cg = (
            cg_in_alert.groupby("airport_alert_id").cumcount(ascending=False) == 0
        )
        df.loc[cg_in_alert_mask, "is_last_lightning_cloud_ground"] = is_last_cg.values

    return df[OUTPUT_COLUMNS]


def split_train_eval(
    df_alerts: pd.DataFrame,
    train_end_year: int = TRAIN_END_YEAR,
    eval_start_year: int = EVAL_START_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporel par année calendaire pour éviter la fuite saisonnière."""
    years = df_alerts["date"].dt.year
    df_train = df_alerts.loc[years <= train_end_year].copy()
    df_eval = df_alerts.loc[years >= eval_start_year].copy()
    return df_train, df_eval


def build_and_save(raw_dir: Path, data_dir: Path) -> None:
    """Construit le dataset segmenté et écrit les CSV global / train / eval."""
    print(f"[1/3] Lecture des CSV bruts depuis {raw_dir}")
    df_lightnings = load_raw_airports(raw_dir)
    print(
        f"      {len(df_lightnings):,} éclairs, "
        f"{df_lightnings['airport'].nunique()} aéroports"
    )

    print(
        f"[2/3] Segmentation (dist <= {MAX_DISTANCE_KM:.0f} km, "
        f"gap <= {MAX_GAP_MINUTES:.0f} min, règle CG+IC)"
    )
    df_alerts = segment_alerts(df_lightnings)
    n_alerts = df_alerts["airport_alert_id"].nunique()
    n_in_alert = df_alerts["airport_alert_id"].notna().sum()
    print(f"      {n_alerts:,} alertes, {n_in_alert:,} éclairs en zone")

    data_dir.mkdir(parents=True, exist_ok=True)
    full_csv = data_dir / "segment_alerts_all_airports.csv"
    train_csv = data_dir / "segment_alerts_all_airports_train.csv"
    eval_csv = data_dir / "segment_alerts_all_airports_eval.csv"

    df_train, df_eval = split_train_eval(df_alerts)

    print(f"[3/3] Écriture dans {data_dir}")
    df_alerts.to_csv(full_csv, index=False)
    df_train.to_csv(train_csv, index=False)
    df_eval.to_csv(eval_csv, index=False)
    print(
        f"      {full_csv.name}   {len(df_alerts):>9,} lignes\n"
        f"      {train_csv.name}  {len(df_train):>9,} lignes "
        f"(<= {TRAIN_END_YEAR})\n"
        f"      {eval_csv.name}   {len(df_eval):>9,} lignes "
        f"(>= {EVAL_START_YEAR})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help=f"Dossier contenant les CSV bruts par aéroport (défaut : {DEFAULT_RAW_DATA_DIR}).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Dossier de sortie pour les CSV segmentés (défaut : {DEFAULT_DATA_DIR}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_and_save(args.raw_dir, args.data_dir)
