# Dataset setup

Raw dataset files are **not** committed to this repository (several are 75MB–2.4GB, and most
carry their own license/terms-of-use that don't permit redistribution). `data/raw/` is
gitignored — download each dataset yourself from its official source and place it as
described below. `configs/tier1.yaml`/`configs/full.yaml` point at these exact paths.

| Dataset | Config name | Place at | Source | Notes |
|---|---|---|---|---|
| MovieLens 100K | `ml100k` | `data/raw/ml100k/u.data` | GroupLens (https://grouplens.org/datasets/movielens/) | Official research dataset, redistribution terms in GroupLens' own license file |
| MovieLens 1M | `ml1m` | `data/raw/ml1m/ratings.dat` | GroupLens (https://grouplens.org/datasets/movielens/) | Same as above |
| Coat | `coat` | `data/raw/coat/train.ascii`, `data/raw/coat/test.ascii` | Schnabel et al.'s Coat shopping dataset (Cornell) | Small (~3MB); search "Coat shopping dataset Schnabel" for the current page |
| MarketBias / Electronics | `electronics` | `data/raw/electronics/df_electronics.csv` | MarketBias dataset release (Amazon Electronics subset) | Check the dataset's own paper/repo for current hosting |
| RentTheRunway | `rentrunway` | `data/raw/rentrunway/renttherunway_final_data.json` | Kaggle "Clothing Fit Dataset for Size Recommendation" (RentTheRunway) | Requires a Kaggle account; follow Kaggle's ToS |
| AMBAR | `ambar` | `data/raw/ambar/ratings_info.csv` (+ `users_info.csv`, `tracks_info.csv`, `artists_info.csv` if you use them elsewhere) | AMBAR dataset's own GitHub release | Check the AMBAR repo for current release assets |
| Last.fm 1K | `lastfm1k` | `data/raw/lastfm1k/userid-timestamp-artid-artname-traid-traname.tsv` | Last.fm 1K Users dataset (Òscar Celma, MTG-UPF) | **2.4GB** — the largest dataset here |
| Amazon Digital Music | `amazon_digital_music` | `data/raw/amazon_digital_music/Digital_Music.jsonl` | Amazon Reviews dataset (UCSD), Digital Music category | Requires accepting the dataset's terms |
| PGPR knowledge graph | `pgpr_kg` | `data/raw/pgpr_kg/ml100k/`, `data/raw/pgpr_kg/ml1m/` (each with `entities/`, `relations/`, `mappings/`, `train.txt`, `test.txt` — see note) | **No verified canonical source found** — see `REGISTRY.md`'s "Known gap" note. This is a research-paper-specific derived artifact, not tracked in the `explanation-quality-recsys` fork's own git history. If you have it from your own prior work, run `rexbench stage-pgpr-kg --source <your existing dir> --dataset ml100k` (or `ml1m`) to copy it into place with validation; otherwise it needs to be re-derived (see `external/explanation-quality-recsys/models/PGPR/preprocess.py`) or sourced from wherever the PGPR paper's authors distribute it. `kg_final.txt`/`e_map.txt`/`r_map.txt`, if present, are leftover artifacts from an unrelated joint-kg/KGAT conversion — verified against the vendored source that PGPR's own training code never reads them, only `entities/`/`relations/`/`mappings/`/`train.txt`/`test.txt` |

## Directory layout expected under `data/raw/`

```
data/raw/
├── ml100k/u.data
├── ml1m/ratings.dat
├── coat/{train,test}.ascii
├── electronics/df_electronics.csv
├── rentrunway/renttherunway_final_data.json
├── ambar/ratings_info.csv
├── lastfm1k/userid-timestamp-artid-artname-traid-traname.tsv
├── amazon_digital_music/Digital_Music.jsonl
└── pgpr_kg/{ml100k,ml1m}/...
```

Each dataset's exact expected filename is also the `raw_path` value in
`configs/tier1.yaml`/`configs/full.yaml` — if you place a file somewhere else, update the
config to match rather than the other way around.