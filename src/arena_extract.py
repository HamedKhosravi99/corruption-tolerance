"""Extract (tstamp, model_a, model_b, winner, anony) from the public Chatbot Arena
battle file clean_battle_20240814_public.json (about 2.1 GB, pretty-printed JSON).

Source: https://storage.googleapis.com/arena_external_data/public/clean_battle_20240814_public.json
The file is parsed line by line (top-level keys sit at an 8-space indent), so the
whole file never has to be held in memory.

Usage: python arena_extract.py <input.json> <output.csv.gz>
"""
import csv
import gzip
import json
import sys

KEYS = ("model_a", "model_b", "winner", "anony", "tstamp")


def records(path):
    rec = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("    {"):
                rec = {}
            elif line.startswith("        \"") and not line.startswith("         "):
                key, _, val = line.strip().rstrip(",").partition(":")
                key = key.strip('"')
                if key in KEYS:
                    rec[key] = json.loads(val)
            elif line.startswith("    }"):
                if all(k in rec for k in KEYS):
                    yield rec


def main(inp, out):
    n = 0
    with gzip.open(out, "wt", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(KEYS)
        for r in records(inp):
            w.writerow([r[k] for k in KEYS])
            n += 1
    print("records written:", n)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
