"""Compare inference risk with and without simulated proxy features."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def simulate_population(size, seed):
    rng = np.random.default_rng(seed)
    age_band = rng.integers(0, 4, size)
    activity = rng.normal(size=size)
    log_odds = -0.7 + 0.25 * age_band + 0.18 * activity
    probability = 1 / (1 + np.exp(-log_odds))
    withheld_attribute = rng.binomial(1, probability)
    baseline = np.column_stack((age_band, activity))
    with_proxies = np.column_stack((
        baseline,
        0.9 * withheld_attribute + rng.normal(scale=0.8, size=size),
        rng.poisson(2 + 1.8 * withheld_attribute),
    ))
    return baseline, with_proxies, withheld_attribute


def main():
    seed = 42
    baseline, with_proxies, outcome = simulate_population(5_000, seed)
    train, test = train_test_split(
        np.arange(outcome.size), test_size=0.3, stratify=outcome, random_state=seed
    )

    print(f"{'Feature set':<22} {'ROC AUC':>8} {'Balanced acc.':>14} {'F1':>8}")
    for name, features in (("Baseline", baseline), ("Baseline + proxies", with_proxies)):
        model = RandomForestClassifier(
            n_estimators=120, min_samples_leaf=12, random_state=seed
        )
        model.fit(features[train], outcome[train])
        probabilities = model.predict_proba(features[test])[:, 1]
        predictions = model.predict(features[test])
        auc = roc_auc_score(outcome[test], probabilities)
        balanced = balanced_accuracy_score(outcome[test], predictions)
        f1 = f1_score(outcome[test], predictions)
        print(f"{name:<22} {auc:8.3f} {balanced:14.3f} {f1:8.3f}")


if __name__ == "__main__":
    main()
