# AutoTab v2 — common tasks. Requires `uv` (https://docs.astral.sh/uv/).

.PHONY: install test lint app predict preprocess train clean

install:            ## create .venv and install the package with app + dev extras
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python -e ".[app,dev]"

test:               ## run unit tests
	.venv/bin/python -m pytest -q

lint:               ## ruff
	.venv/bin/ruff check autotab tests streamlit

app:                ## launch the Streamlit UI
	.venv/bin/streamlit run streamlit/autotab_app.py

FILE ?= data/GuitarSet/audio/audio_mic/00_BN1-129-Eb_solo_mic.wav
predict:            ## transcribe FILE=<wav> with the pretrained weights
	.venv/bin/autotab predict "$(FILE)" --mode simple

preprocess:         ## build npz representations for every GuitarSet file (JOBS=n workers)
	.venv/bin/autotab preprocess --mode c -j $(or $(JOBS),4)

train:              ## 6-fold cross-validation on data/spec_repr/c
	.venv/bin/autotab train --mode c --epochs 8 --id-file ''

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
