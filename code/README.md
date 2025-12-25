# Analysis Code

## Requirements

```
numpy
pandas
scipy
```

## Scripts

### 01_descriptive.py
Computes basic statistics:
- Distribution of P(A>B) values
- Win-rate per outcome
- Preference clarity metrics

### 02_thurstonian.py
Fits the Thurstonian utility model:
- Maximum Likelihood Estimation
- Computes U(o) for each outcome
- Model accuracy metrics

### 03_coherence.py
Analyzes preference coherence:
- Transitivity analysis
- Cycle detection
- Completeness metrics

## Usage

```bash
python 01_descriptive.py --input ../data/preferences.json
python 02_thurstonian.py --input ../data/preferences.json
python 03_coherence.py --input ../data/preferences.json
```

## Note

These scripts were adapted from the original study code. Path configurations may need adjustment for your environment.
