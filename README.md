# Collective Inference Risk Lab

A small, reproducible Python experiment accompanying the CIRL dashboard concept. It compares a classifier using baseline variables with one that also receives deliberately correlated proxies. The outcome is a randomly simulated binary attribute. No real or personal records are used.

## Run

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiment.py
```

Both models use the same stratified train/test split. Results include ROC AUC, balanced accuracy, and F1. The experiment demonstrates how synthetic proxy features can increase inferability in a controlled setting. The scores depend on the simulation assumptions.
