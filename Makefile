LOG_DIR := logs
OUTPUT_DIR := data/processed

.PHONY: all | log_dir
all: splits

.PHONY: clean
clean:
	rm -r $(OUTPUT_DIR) $(LOG_DIR)

.PHONY: splits
splits: cluster
	python scripts/make_splits.py &> $(LOG_DIR)/$@.log

.PHONY: cluster
cluster: fastas
	bash scripts/cluster.sh &> $(LOG_DIR)/$@.log

.PHONY: fastas
fastas:
	python scripts/make_fastas.py &> $(LOG_DIR)/$@.log

.PHONY: log_dir
log_dir:
	mkdir -p $(LOG_DIR)