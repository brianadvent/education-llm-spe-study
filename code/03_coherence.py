"""
SPE-Studie: Kohärenz-Metriken

Dieses Skript berechnet Kohärenzmetriken aus dem Paper:
1. Transitivität: Wenn A>B und B>C, dann A>C?
2. Completeness: Wie entschieden ist das Modell?
3. Zyklen-Analyse: Wie viele intransitive Tripel gibt es?

Aufruf:
    uv run python -m src.analysis.03_coherence [--input FILE]

Output:
    results/03_coherence.json
    results/03_intransitive_triplets.csv
"""

import argparse
import json
import random
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

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


def get_preference(preferences: Dict[str, float], id_a: str, id_b: str) -> float:
    """Holt P(A>B), auch wenn nur P(B>A) gespeichert ist."""
    key1 = f"{id_a}|{id_b}"
    key2 = f"{id_b}|{id_a}"

    if key1 in preferences:
        return preferences[key1]
    elif key2 in preferences:
        return 1 - preferences[key2]
    else:
        return np.nan


def check_transitivity(p_ab: float, p_bc: float, p_ac: float, threshold: float = 0.5) -> str:
    """
    Prüft Transitivität für ein Tripel.

    Wenn A>B (p_ab > threshold) und B>C (p_bc > threshold),
    dann sollte A>C (p_ac > threshold).

    Returns:
        'transitive': Konsistent
        'intransitive': Verletzung (Zyklus)
        'unclear': Mindestens eine Präferenz unklar
    """
    a_beats_b = p_ab > threshold
    b_beats_c = p_bc > threshold
    a_beats_c = p_ac > threshold

    # Prüfe alle möglichen Zyklen
    # Zyklus 1: A>B, B>C, aber C>A (also nicht A>C)
    if a_beats_b and b_beats_c and not a_beats_c:
        return 'intransitive'

    # Zyklus 2: B>A, C>B, aber A>C
    b_beats_a = p_ab < threshold
    c_beats_b = p_bc < threshold

    if b_beats_a and c_beats_b and a_beats_c:
        return 'intransitive'

    # Weitere Zyklusvarianten
    if a_beats_b and not b_beats_c and not a_beats_c:
        # A>B, C>B, C>A - kein Zyklus
        pass

    return 'transitive'


def compute_transitivity_metrics(
    preferences: Dict[str, float],
    outcome_ids: List[str],
    sample_size: int = None,
    threshold: float = 0.5
) -> Dict:
    """
    Berechnet Transitivitätsmetriken über alle Tripel.

    Args:
        preferences: Präferenzdaten
        outcome_ids: Liste der Outcome-IDs
        sample_size: Wenn gesetzt, nur Stichprobe analysieren
        threshold: Schwelle für "A bevorzugt B"

    Returns:
        Metriken-Dictionary
    """
    all_triplets = list(combinations(outcome_ids, 3))

    if sample_size and len(all_triplets) > sample_size:
        triplets = random.sample(all_triplets, sample_size)
        sampled = True
    else:
        triplets = all_triplets
        sampled = False

    transitive = 0
    intransitive = 0
    unclear = 0
    intransitive_examples = []

    for a, b, c in triplets:
        p_ab = get_preference(preferences, a, b)
        p_bc = get_preference(preferences, b, c)
        p_ac = get_preference(preferences, a, c)

        if np.isnan(p_ab) or np.isnan(p_bc) or np.isnan(p_ac):
            unclear += 1
            continue

        result = check_transitivity(p_ab, p_bc, p_ac, threshold)

        if result == 'transitive':
            transitive += 1
        elif result == 'intransitive':
            intransitive += 1
            if len(intransitive_examples) < 100:  # Nur erste 100 speichern
                intransitive_examples.append({
                    'a': a, 'b': b, 'c': c,
                    'p_ab': p_ab, 'p_bc': p_bc, 'p_ac': p_ac
                })

    total_valid = transitive + intransitive

    return {
        'total_triplets': len(triplets),
        'sampled': sampled,
        'transitive': transitive,
        'intransitive': intransitive,
        'unclear': unclear,
        'transitivity_rate': transitive / total_valid if total_valid > 0 else 0,
        'intransitivity_rate': intransitive / total_valid if total_valid > 0 else 0,
        'examples': intransitive_examples
    }


def compute_completeness(preferences: Dict[str, float]) -> Dict:
    """
    Berechnet Completeness-Metrik.

    Completeness = Wie "entschieden" ist das Modell?
    Gemessen als durchschnittliche Distanz von 0.5.
    """
    values = list(preferences.values())

    # Distanz von 0.5 (Indifferenz)
    distances = [abs(v - 0.5) for v in values]

    # Anteil klarer Präferenzen (>0.7 oder <0.3)
    clear = sum(1 for v in values if v > 0.7 or v < 0.3)

    return {
        'mean_distance_from_indifference': np.mean(distances),
        'median_distance_from_indifference': np.median(distances),
        'std_distance': np.std(distances),
        'clear_preferences_rate': clear / len(values),
        'n_pairs': len(values)
    }


def compute_consistency(preferences: Dict[str, float], k: int) -> Dict:
    """
    Berechnet Konsistenz basierend auf K Wiederholungen.

    Bei K=10 Wiederholungen zeigen Werte nahe 0.5 Inkonsistenz,
    während 0.0 oder 1.0 perfekte Konsistenz bedeuten.
    """
    values = list(preferences.values())

    # Erwartete Varianz bei zufälligem Raten: 0.25 (Binomial)
    # Beobachtete "Varianz" als Distanz von Extremen
    consistency_scores = [min(v, 1 - v) * 2 for v in values]  # 0 = konsistent, 1 = inkonsistent

    return {
        'mean_inconsistency': np.mean(consistency_scores),
        'highly_consistent_rate': sum(1 for s in consistency_scores if s < 0.2) / len(values),
        'highly_inconsistent_rate': sum(1 for s in consistency_scores if s > 0.8) / len(values)
    }


def main():
    parser = argparse.ArgumentParser(description='Kohärenz-Analyse der SPE-Präferenzdaten')
    parser.add_argument('--input', '-i', type=Path,
                        help='Pfad zur Präferenzdatei (default: neueste)')
    parser.add_argument('--sample', '-s', type=int, default=10000,
                        help='Stichprobengröße für Tripel-Analyse (default: 10000)')
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
    print("SPE-Studie: Kohärenz-Analyse")
    print("=" * 60)
    print(f"Input: {input_file}")
    print()

    # Daten laden
    preferences, meta = load_preferences(input_file)
    outcomes = load_outcomes()
    outcome_ids = sorted(outcomes.keys())
    k = meta.get('k_repetitions', 10)

    print(f"Paare: {len(preferences):,}")
    print(f"Outcomes: {len(outcome_ids)}")
    print(f"K-Wiederholungen: {k}")
    print()

    # 1. Transitivität
    print("1. Transitivitäts-Analyse")
    print("-" * 40)
    n_triplets = len(list(combinations(outcome_ids, 3)))
    print(f"  Mögliche Tripel: {n_triplets:,}")

    transitivity = compute_transitivity_metrics(
        preferences, outcome_ids,
        sample_size=args.sample
    )

    print(f"  Analysierte Tripel: {transitivity['total_triplets']:,}")
    print(f"  {'(Stichprobe)' if transitivity['sampled'] else '(vollständig)'}")
    print(f"  Transitiv: {transitivity['transitive']:,} ({transitivity['transitivity_rate']:.1%})")
    print(f"  Intransitiv: {transitivity['intransitive']:,} ({transitivity['intransitivity_rate']:.1%})")
    print()

    # 2. Completeness
    print("2. Completeness (Entschiedenheit)")
    print("-" * 40)
    completeness = compute_completeness(preferences)
    print(f"  Mittlere Distanz von 0.5: {completeness['mean_distance_from_indifference']:.3f}")
    print(f"  Anteil klarer Präferenzen (>0.7 oder <0.3): {completeness['clear_preferences_rate']:.1%}")
    print()

    # 3. Konsistenz
    print("3. Konsistenz (basierend auf K={k} Wiederholungen)")
    print("-" * 40)
    consistency = compute_consistency(preferences, k)
    print(f"  Mittlere Inkonsistenz: {consistency['mean_inconsistency']:.3f}")
    print(f"  Hoch konsistent (<0.2): {consistency['highly_consistent_rate']:.1%}")
    print(f"  Hoch inkonsistent (>0.8): {consistency['highly_inconsistent_rate']:.1%}")
    print()

    # Beispiele für intransitive Tripel
    if transitivity['examples']:
        print("4. Beispiele für intransitive Tripel (Zyklen)")
        print("-" * 40)
        for i, ex in enumerate(transitivity['examples'][:5], 1):
            print(f"  {i}. {ex['a']} > {ex['b']} ({ex['p_ab']:.0%})")
            print(f"     {ex['b']} > {ex['c']} ({ex['p_bc']:.0%})")
            print(f"     {ex['c']} > {ex['a']} ({1-ex['p_ac']:.0%}) <- Zyklus!")
            print()

    # Ergebnisse speichern
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    results = {
        'meta': {
            'analysis': '03_coherence',
            'timestamp': datetime.now().isoformat(),
            'input_file': str(input_file),
            'source_meta': meta
        },
        'transitivity': {
            'total_triplets': transitivity['total_triplets'],
            'sampled': transitivity['sampled'],
            'transitive': transitivity['transitive'],
            'intransitive': transitivity['intransitive'],
            'transitivity_rate': transitivity['transitivity_rate']
        },
        'completeness': completeness,
        'consistency': consistency
    }

    json_file = RESULTS_DIR / '03_coherence.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Kohärenz-Metriken gespeichert: {json_file}")

    # CSV mit intransitiven Tripeln
    if transitivity['examples']:
        df = pd.DataFrame(transitivity['examples'])
        csv_file = RESULTS_DIR / '03_intransitive_triplets.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Intransitive Tripel gespeichert: {csv_file}")

    print()
    print("=" * 60)
    print("Kohärenz-Analyse abgeschlossen!")


if __name__ == '__main__':
    main()
