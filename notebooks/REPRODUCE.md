# Reproduire l'entraînement V3 (multi-GPU)

Code à coller dans un notebook d'un environnement distant disposant d'un ou
plusieurs GPU. Le code vit dans le dépôt — on le clone puis on l'importe, au
lieu de tout recopier dans des cellules.

## Prérequis

- Accélérateur **GPU** activé (idéalement ≥ 2 GPU pour la parallélisation).
- Le CSV d'entraînement segmenté (`segment_alerts_all_airports_train.csv`)
  disponible dans l'environnement. Il se génère depuis les éclairs bruts avec
  `build_data.py` (voir racine du dépôt).

## Cellule 1 — récupérer le code

```python
# Clone au premier lancement, sinon met à jour (git pull) pour récupérer tes
# dernières modifs poussées depuis VSCode.
!git clone -b reproduction https://github.com/Lucas64000/prediction-orage.git code \
    2>/dev/null || (cd code && git pull)

import sys
sys.path.insert(0, "code/src")
```

## Cellule 2 — lancer

```python
import train_parallel

results = train_parallel.run_v3(
    data_csv="<chemin vers segment_alerts_all_airports_train.csv>",
    output_dir="outputs",
    n_epochs=80,
)
```

`run_v3` détecte automatiquement les GPU, répartit les variantes (V3a GRU,
V3b/V3c Transformer, V3d par aéroport) — une par GPU — puis écrit dans
`outputs/` : les poids (`models/*.pt`), le résumé (`phase2bis_v3_results.json`)
et les erreurs détaillées (`errors_v3_*.parquet`).

## Workflow itératif

1. Modifier le code dans VSCode (ex. `src/neural_hawkes_v3.py`).
2. `git push` sur la branche `reproduction`.
3. Dans le notebook : ré-exécuter la **cellule 1** (le `git pull` récupère la
   modif), puis redémarrer le kernel pour recharger les modules, et relancer la
   **cellule 2**.

## Lancer en local (CLI)

```bash
python src/train_parallel.py --data-csv data/segment_alerts_all_airports_train.csv --output-dir outputs
```

La run mono-GPU d'origine des étudiants reste disponible via
`python src/neural_hawkes_v3.py`.
