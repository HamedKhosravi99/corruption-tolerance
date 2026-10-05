# Shortcuts for the commands of the README. Run from the repository root.
PY ?= python3

.PHONY: paper figures tables experiments arena network all clean-pyc

## Rebuild the paper's figures and tables from the saved results (about a minute).
paper: figures tables

figures:
	$(PY) scripts/make_figures.py

# make_tables.py reads results/e4b_table.csv, which make_figures.py writes.
tables: figures
	$(PY) scripts/make_tables.py

## Rerun the experiments that need neither the Arena data nor the network (several hours in total).
experiments:
	$(PY) experiments/e1_validity.py      | tee results/logs/e1_log.txt
	$(PY) experiments/e2_rate.py          | tee results/logs/e2_log.txt
	$(PY) experiments/e3_selection.py     | tee results/logs/e3_log.txt
	$(PY) experiments/e5_regime.py --figure | tee results/logs/e5_log.txt
	$(PY) experiments/e7_benchmark.py a b fig | tee results/logs/e7_log.txt
	$(PY) experiments/e8_frontier.py a c b  | tee results/logs/e8_log.txt
	$(PY) experiments/e8_swebench.py a c b  | tee results/logs/e8_swebench_log.txt
	$(PY) experiments/e9_average_rate.py  | tee results/logs/e9_log.txt
	$(PY) experiments/e10_breakdown.py    | tee results/logs/e10_log.txt

## Experiments on the Chatbot Arena votes (needs data/arena/arena_battles.csv.gz or ARENA_DATA).
arena:
	$(PY) experiments/e4_arena.py stats   | tee results/logs/e4_data_log.txt
	$(PY) experiments/e4_arena.py a call  | tee results/logs/e4ac_log.txt
	$(PY) experiments/e4_arena.py b       | tee results/logs/e4b_log.txt
	$(PY) experiments/e6_suppress.py      | tee results/logs/e6_log.txt

## Download the benchmark data again (the copies in data/ are the ones used in the paper).
network:
	$(PY) experiments/e7_benchmark.py fetch
	$(PY) experiments/e8_frontier.py fetch
	$(PY) experiments/e8_swebench.py fetch

all: arena experiments paper

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
