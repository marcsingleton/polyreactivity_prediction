.PHONY: all
all: splits

.PHONY: splits
splits: cluster
	python scripts/make_splits.py

.PHONY: cluster
cluster: fastas
	bash scripts/cluster.sh

.PHONY: fastas
fastas:
	python scripts/make_fastas.py
