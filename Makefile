# Adaptive Reasoner — dev tasks.
# Run `make test` to run the offline verification suite in sequence.

PY ?= python3

.PHONY: test data train train-smoke eval

# Offline verification suite (tiny random GPT-2, no download, no GPU needed).
# Stops on the first failing script (each asserts internally).
test:
	$(PY) tests/verify_grad.py
	$(PY) tests/check_forward.py
	$(PY) tests/check_generate.py

# Regenerate datasets/reasoning_train.json (240 synthetic rows).
data:
	$(PY) datasets/dataset.py

# Full training run (use a GPU machine; CPU is ~1 step/min).
train:
	$(PY) train_reasoner.py --epochs 3 --batch 8

# Quick CPU-friendly smoke test.
train-smoke:
	$(PY) train_reasoner.py --limit 32 --epochs 1 --batch 4

# Evaluate the trained checkpoint against baseline Qwen.
eval:
	$(PY) evaluate_reasoner.py --checkpoint checkpoints/adaptive_final.pt
