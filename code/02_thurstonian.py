"""
SPE-Studie: Thurstonian Utility Model

Dieses Skript implementiert das Thurstonian Model aus dem Paper:
- Schätzt latente Utility-Werte U(o) für jedes Outcome
- Verwendet Maximum Likelihood Estimation (MLE)
- Berechnet Model Fit (Accuracy)

Theorie:
    U(o) ~ N(μ(o), σ²)
    P(x ≻ y) = Φ((μ(x) - μ(y)) / √(σ²(x) + σ²(y)))

    Bei homogener Varianz (σ² = const):
    P(x ≻ y) = Φ((μ(x) - μ(y)) / (σ√2))

Aufruf:
    uv run python -m src.analysis.02_thurstonian [--input FILE]

Output:
    results/02_utilities.json
    results/02_utilities.csv
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# Pfade
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTCOMES_FILE = DATA_DIR / "outcomes" / "outcomes.json"


def load_preferences(filepath: Path) -> Tuple[Dict[str, float], dict]:
    """Lädt Präferenzdaten aus JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['preferences'], data['meta']


def load_outcomes() -> Dict[str, dict]:
    """Lädt Outcome-Texte und Metadaten."""
    with open(OUTCOMES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {o['id']: o for o in data['outcomes']}


def preferences_to_matrix(preferences: Dict[str, float], outcome_ids: List[str]) -> np.ndarray:
    """
    Konvertiert Präferenzen in eine Matrix.

    Returns:
        Matrix P wo P[i,j] = P(outcome_i > outcome_j)
    """
    n = len(outcome_ids)
    id_to_idx = {oid: i for i, oid in enumerate(outcome_ids)}
    P = np.full((n, n), np.nan)

    for pair_key, p_val in preferences.items():
        id_a, id_b = pair_key.split('|')
        if id_a in id_to_idx and id_b in id_to_idx:
            i, j = id_to_idx[id_a], id_to_idx[id_b]
            P[i, j] = p_val
            P[j, i] = 1 - p_val  # Symmetrie

    return P


def thurstonian_log_likelihood(mu: np.ndarray, P: np.ndarray, sigma: float = 1.0) -> float:
    """
    Berechnet die negative Log-Likelihood für das Thurstonian Model.

    Args:
        mu: Utility-Werte (n,)
        P: Präferenzmatrix (n, n)
        sigma: Standardabweichung (konstant)

    Returns:
        Negative Log-Likelihood (zu minimieren)
    """
    n = len(mu)
    ll = 0.0
    scale = sigma * np.sqrt(2)

    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(P[i, j]):
                # P(i > j) = Φ((μ_i - μ_j) / (σ√2))
                p_pred = norm.cdf((mu[i] - mu[j]) / scale)

                # Clip um log(0) zu vermeiden
                p_pred = np.clip(p_pred, 1e-10, 1 - 1e-10)
                p_obs = np.clip(P[i, j], 1e-10, 1 - 1e-10)

                # Binary Cross-Entropy
                ll -= p_obs * np.log(p_pred) + (1 - p_obs) * np.log(1 - p_pred)

    return ll


def fit_thurstonian_model(P: np.ndarray, max_iter: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Fittet das Thurstonian Model mit MLE.

    Args:
        P: Präferenzmatrix (n, n)
        max_iter: Maximale Iterationen

    Returns:
        mu: Geschätzte Utility-Werte
        sigma: Geschätzte Standardabweichung
    """
    n = P.shape[0]

    # Initialisierung: Win-Rate als Startpunkt
    win_rates = np.nanmean(P, axis=1)
    mu_init = norm.ppf(np.clip(win_rates, 0.01, 0.99))

    # Zentrieren (Identifikation: mean(μ) = 0)
    mu_init = mu_init - np.mean(mu_init)

    print(f"  Starte Optimierung mit {n} Outcomes...")

    # Optimierung
    result = minimize(
        lambda mu: thurstonian_log_likelihood(mu - np.mean(mu), P),
        mu_init,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': False}
    )

    mu_fitted = result.x - np.mean(result.x)  # Zentrieren

    print(f"  Optimierung abgeschlossen: {result.success}")
    print(f"  Finale Log-Likelihood: {-result.fun:.2f}")

    return mu_fitted, 1.0  # sigma = 1 (normiert)


def compute_model_accuracy(mu: np.ndarray, P: np.ndarray) -> Dict[str, float]:
    """
    Berechnet die Vorhersagegenauigkeit des Modells.

    Returns:
        accuracy: Anteil korrekt vorhergesagter Präferenzen
        mean_abs_error: Mittlerer absoluter Fehler
    """
    n = len(mu)
    correct = 0
    total = 0
    abs_errors = []

    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(P[i, j]):
                # Vorhersage: i > j wenn μ_i > μ_j
                p_pred = norm.cdf((mu[i] - mu[j]) / np.sqrt(2))
                p_obs = P[i, j]

                # Korrekt wenn beide > 0.5 oder beide < 0.5
                pred_i_wins = p_pred > 0.5
                obs_i_wins = p_obs > 0.5

                if pred_i_wins == obs_i_wins:
                    correct += 1
                total += 1

                abs_errors.append(abs(p_pred - p_obs))

    return {
        'accuracy': correct / total if total > 0 else 0,
        'n_pairs': total,
        'n_correct': correct,
        'mean_abs_error': np.mean(abs_errors),
        'median_abs_error': np.median(abs_errors)
    }


def main():
    parser = argparse.ArgumentParser(description='Thurstonian Model für SPE-Präferenzdaten')
    parser.add_argument('--input', '-i', type=Path,
                        help='Pfad zur Präferenzdatei (default: neueste)')
    args = parser.parse_args()

    # Input-Datei finden
    if args.input:
        input_file = args.input
    else:
        pref_files = sorted(DATA_DIR.glob('preferences/preferences_*.json'))
        if not pref_files:
            print("Keine Präferenzdatei gefunden!")
            return
        input_file = pref_files[-1]

    print("=" * 60)
    print("SPE-Studie: Thurstonian Utility Model")
    print("=" * 60)
    print(f"Input: {input_file}")
    print()

    # Daten laden
    preferences, meta = load_preferences(input_file)
    outcomes = load_outcomes()
    outcome_ids = sorted(outcomes.keys())

    print(f"Paare: {len(preferences):,}")
    print(f"Outcomes: {len(outcome_ids)}")
    print()

    # Präferenzmatrix erstellen
    print("1. Erstelle Präferenzmatrix...")
    P = preferences_to_matrix(preferences, outcome_ids)
    print(f"  Matrix-Shape: {P.shape}")
    print(f"  Nicht-NaN Einträge: {np.sum(~np.isnan(P)):,}")
    print()

    # Thurstonian Model fitten
    print("2. Fitte Thurstonian Model (MLE)...")
    mu, sigma = fit_thurstonian_model(P)
    print()

    # Model Accuracy
    print("3. Berechne Model Fit...")
    accuracy = compute_model_accuracy(mu, P)
    print(f"  Accuracy: {accuracy['accuracy']:.1%}")
    print(f"  Mean Absolute Error: {accuracy['mean_abs_error']:.3f}")
    print()

    # Utilities sortieren
    utility_ranking = sorted(
        [(outcome_ids[i], mu[i]) for i in range(len(mu))],
        key=lambda x: x[1],
        reverse=True
    )

    print("4. Utility Rankings")
    print("-" * 40)
    print("\nTOP 10 (höchste Utility):")
    for i, (oid, u) in enumerate(utility_ranking[:10], 1):
        text = outcomes.get(oid, {}).get('text', 'N/A')[:50]
        print(f"  {i:2}. [{oid}] U={u:+.3f} - {text}...")

    print("\nBOTTOM 10 (niedrigste Utility):")
    for i, (oid, u) in enumerate(utility_ranking[-10:], 1):
        text = outcomes.get(oid, {}).get('text', 'N/A')[:50]
        print(f"  {i:2}. [{oid}] U={u:+.3f} - {text}...")
    print()

    # Ergebnisse speichern
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    results = {
        'meta': {
            'analysis': '02_thurstonian',
            'timestamp': datetime.now().isoformat(),
            'input_file': str(input_file),
            'source_meta': meta
        },
        'model': {
            'sigma': sigma,
            'n_outcomes': len(outcome_ids),
            'n_pairs': len(preferences)
        },
        'accuracy': accuracy,
        'utilities': {oid: float(u) for oid, u in utility_ranking}
    }

    json_file = RESULTS_DIR / '02_utilities.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Utilities (JSON) gespeichert: {json_file}")

    # CSV
    csv_data = []
    for rank, (oid, u) in enumerate(utility_ranking, 1):
        outcome_info = outcomes.get(oid, {})
        csv_data.append({
            'rank': rank,
            'outcome_id': oid,
            'utility': u,
            'item': outcome_info.get('item', ''),
            'section': oid.split('_')[0],
            'text': outcome_info.get('text', '')[:150]
        })

    df = pd.DataFrame(csv_data)
    csv_file = RESULTS_DIR / '02_utilities.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Utilities (CSV) gespeichert: {csv_file}")

    print()
    print("=" * 60)
    print("Thurstonian Model abgeschlossen!")


if __name__ == '__main__':
    main()
