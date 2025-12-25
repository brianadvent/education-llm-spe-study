"""
SPE-Studie: Deskriptive Analyse der Präferenzdaten

Dieses Skript berechnet:
1. Verteilung der P(A>B) Werte
2. Win-Rate pro Outcome
3. Top/Bottom Rankings
4. Grundlegende Statistiken

Aufruf:
    uv run python -m src.analysis.01_descriptive [--input FILE]

Output:
    results/01_descriptive_stats.json
    results/01_outcome_rankings.csv
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Pfade
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTCOMES_FILE = DATA_DIR / "outcomes" / "outcomes.json"


def load_preferences(filepath: Path) -> Dict[str, float]:
    """Lädt Präferenzdaten aus JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['preferences'], data['meta']


def load_outcomes() -> Dict[str, dict]:
    """Lädt Outcome-Texte und Metadaten."""
    with open(OUTCOMES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {o['id']: o for o in data['outcomes']}


def compute_distribution(preferences: Dict[str, float]) -> Dict[str, int]:
    """Berechnet die Verteilung der P(A>B) Werte."""
    bins = {
        '0.0': 0, '0.1': 0, '0.2': 0, '0.3': 0, '0.4': 0,
        '0.5': 0, '0.6': 0, '0.7': 0, '0.8': 0, '0.9': 0, '1.0': 0
    }

    for p in preferences.values():
        # Runde auf 1 Dezimalstelle für Binning
        bin_key = f"{round(p, 1):.1f}"
        if bin_key in bins:
            bins[bin_key] += 1

    return bins


def compute_win_rates(preferences: Dict[str, float]) -> Dict[str, Dict]:
    """
    Berechnet Win-Rate für jedes Outcome.

    Win = P(A>B) > 0.5 für Outcome A
    Loss = P(A>B) < 0.5 für Outcome A (= Win für B)
    """
    wins = defaultdict(int)
    losses = defaultdict(int)
    ties = defaultdict(int)
    total = defaultdict(int)

    for pair_key, p_a in preferences.items():
        id_a, id_b = pair_key.split('|')

        total[id_a] += 1
        total[id_b] += 1

        if p_a > 0.5:
            wins[id_a] += 1
            losses[id_b] += 1
        elif p_a < 0.5:
            losses[id_a] += 1
            wins[id_b] += 1
        else:  # p_a == 0.5
            ties[id_a] += 1
            ties[id_b] += 1

    # Win-Rate berechnen
    results = {}
    for outcome_id in total:
        results[outcome_id] = {
            'wins': wins[outcome_id],
            'losses': losses[outcome_id],
            'ties': ties[outcome_id],
            'total': total[outcome_id],
            'win_rate': wins[outcome_id] / total[outcome_id] if total[outcome_id] > 0 else 0
        }

    return results


def compute_preference_clarity(preferences: Dict[str, float]) -> Dict[str, float]:
    """
    Berechnet Klarheitsmetriken der Präferenzen.

    - clear_a: P(A>B) >= 0.9 (A klar bevorzugt)
    - clear_b: P(A>B) <= 0.1 (B klar bevorzugt)
    - unclear: 0.4 <= P(A>B) <= 0.6 (unklar)
    """
    values = list(preferences.values())
    n = len(values)

    clear_a = sum(1 for v in values if v >= 0.9)
    clear_b = sum(1 for v in values if v <= 0.1)
    unclear = sum(1 for v in values if 0.4 <= v <= 0.6)

    return {
        'clear_a_count': clear_a,
        'clear_a_pct': clear_a / n,
        'clear_b_count': clear_b,
        'clear_b_pct': clear_b / n,
        'unclear_count': unclear,
        'unclear_pct': unclear / n,
        'total_clear': clear_a + clear_b,
        'total_clear_pct': (clear_a + clear_b) / n
    }


def main():
    parser = argparse.ArgumentParser(description='Deskriptive Analyse der SPE-Präferenzdaten')
    parser.add_argument('--input', '-i', type=Path,
                        help='Pfad zur Präferenzdatei (default: neueste in data/preferences/)')
    args = parser.parse_args()

    # Input-Datei finden
    if args.input:
        input_file = args.input
    else:
        pref_files = sorted(DATA_DIR.glob('preferences/preferences_*.json'))
        if not pref_files:
            print("Keine Präferenzdatei gefunden!")
            return
        input_file = pref_files[-1]  # Neueste Datei

    print("=" * 60)
    print("SPE-Studie: Deskriptive Analyse")
    print("=" * 60)
    print(f"Input: {input_file}")
    print()

    # Daten laden
    preferences, meta = load_preferences(input_file)
    outcomes = load_outcomes()

    print(f"Paare geladen: {len(preferences):,}")
    print(f"Outcomes: {len(outcomes)}")
    print(f"Modell: {meta.get('model', 'unbekannt')}")
    print(f"K-Wiederholungen: {meta.get('k_repetitions', 'unbekannt')}")
    print()

    # 1. Verteilung berechnen
    print("1. Verteilung der P(A>B) Werte")
    print("-" * 40)
    distribution = compute_distribution(preferences)
    for bin_label, count in distribution.items():
        pct = count / len(preferences) * 100
        bar = '█' * int(pct / 2)
        print(f"  {bin_label}: {count:5} ({pct:5.1f}%) {bar}")
    print()

    # 2. Präferenzklarheit
    print("2. Präferenzklarheit")
    print("-" * 40)
    clarity = compute_preference_clarity(preferences)
    print(f"  A klar bevorzugt (≥0.9): {clarity['clear_a_count']:,} ({clarity['clear_a_pct']:.1%})")
    print(f"  B klar bevorzugt (≤0.1): {clarity['clear_b_count']:,} ({clarity['clear_b_pct']:.1%})")
    print(f"  Unklar (0.4-0.6):        {clarity['unclear_count']:,} ({clarity['unclear_pct']:.1%})")
    print(f"  Gesamt klar:             {clarity['total_clear']:,} ({clarity['total_clear_pct']:.1%})")
    print()

    # 3. Win-Rates berechnen
    print("3. Win-Rate Rankings")
    print("-" * 40)
    win_rates = compute_win_rates(preferences)
    sorted_outcomes = sorted(win_rates.items(), key=lambda x: x[1]['win_rate'], reverse=True)

    print("\nTOP 10 (am meisten bevorzugt):")
    for i, (oid, stats) in enumerate(sorted_outcomes[:10], 1):
        text = outcomes.get(oid, {}).get('text', 'N/A')[:60]
        print(f"  {i:2}. [{oid}] {stats['win_rate']:.1%} - {text}...")

    print("\nBOTTOM 10 (am wenigsten bevorzugt):")
    for i, (oid, stats) in enumerate(sorted_outcomes[-10:], 1):
        text = outcomes.get(oid, {}).get('text', 'N/A')[:60]
        print(f"  {i:2}. [{oid}] {stats['win_rate']:.1%} - {text}...")
    print()

    # Ergebnisse speichern
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON mit Statistiken
    stats_output = {
        'meta': {
            'analysis': '01_descriptive',
            'timestamp': datetime.now().isoformat(),
            'input_file': str(input_file),
            'source_meta': meta
        },
        'distribution': distribution,
        'clarity': clarity,
        'summary': {
            'n_pairs': len(preferences),
            'n_outcomes': len(outcomes),
            'mean_preference': sum(preferences.values()) / len(preferences),
            'median_preference': sorted(preferences.values())[len(preferences) // 2]
        }
    }

    stats_file = RESULTS_DIR / '01_descriptive_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    print(f"Statistiken gespeichert: {stats_file}")

    # CSV mit Rankings
    rankings_data = []
    for oid, stats in sorted_outcomes:
        outcome_info = outcomes.get(oid, {})
        rankings_data.append({
            'outcome_id': oid,
            'item': outcome_info.get('item', ''),
            'section': oid.split('_')[0],
            'win_rate': stats['win_rate'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'ties': stats['ties'],
            'total': stats['total'],
            'text': outcome_info.get('text', '')[:100]
        })

    df = pd.DataFrame(rankings_data)
    rankings_file = RESULTS_DIR / '01_outcome_rankings.csv'
    df.to_csv(rankings_file, index=False, encoding='utf-8')
    print(f"Rankings gespeichert: {rankings_file}")

    print()
    print("=" * 60)
    print("Analyse abgeschlossen!")


if __name__ == '__main__':
    main()
