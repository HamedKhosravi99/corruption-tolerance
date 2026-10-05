# Chatbot Arena votes (not included)

E4 and E6 use the public Chatbot Arena battle file `clean_battle_20240814_public.json`
(about 2.1 GB; 1,799,991 anonymous votes among 129 models, 2023-04-24 to 2024-08-14).
The file states no license, so it is not redistributed here. To reproduce E4 and E6:

```bash
curl -o clean_battle_20240814_public.json \
  https://storage.googleapis.com/arena_external_data/public/clean_battle_20240814_public.json
python src/arena_extract.py clean_battle_20240814_public.json data/arena/arena_battles.csv.gz
```

The extraction keeps the model names, the winner, the time stamp and the anonymity flag, and
takes about half a minute. The scripts look for `data/arena/arena_battles.csv.gz`; to keep the
extract elsewhere, set `ARENA_DATA` to its path. `python experiments/e4_arena.py stats` then
prints the counts quoted in the paper.

The saved results of E4 and E6 are in `results/`, so the paper's figures and tables can be
rebuilt without these data.
