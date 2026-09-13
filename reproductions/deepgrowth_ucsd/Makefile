.PHONY: test synthetic-baselines preflight

test:
	python -m unittest discover -s tests -v
	python -m compileall -q ucsd_repro scripts

synthetic-baselines:
	python scripts/make_synthetic.py --output-dir data/synthetic
	python scripts/evaluate_baselines.py --metadata data/synthetic/metadata.csv --output-dir outputs/synthetic-baselines

preflight:
	python scripts/preflight.py --require-cuda
