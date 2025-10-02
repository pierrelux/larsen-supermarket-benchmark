# ECC'07 Supermarket Refrigeration Benchmark

Implementation of the ECC'07 supermarket refrigeration benchmark from Larsen et al. (2007). 

Note: not all details are present for a perfect reproduction of the figures shown in the paper. For example, the capacity quantization and unit dispatch policy (midpoint thresholds and optional lead/lag alternation), the exact switch-counting convention, the initial conditions (Psuc and case states), and the VFD min-capacity behavior are not fully specified in the paper and are therefore defined explicitly here, along with small numerical guards (anti-windup, dρ/dP floor, optional load noise).

Larsen, L. F. S., Izadi-Zamanabadi, R., & Wisniewski, R. (2007, July 2-5). Supermarket Refrigeration System - Benchmark for Hybrid System Control. *Proceedings of the European Control Conference 2007*, Kos, Greece. TuA03.5.

## Setup

```bash
# Clone repository
git clone git@github.com:pierrelux/larsen-supermarket-benchmark.git
cd larsen-supermarket-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib

# Run benchmark
python supermarket.py
```

## Usage

Default configuration runs clean baseline:
```bash
python supermarket.py
```

Enable baseline alignment features by editing flags in `supermarket.py`:
```python
ENABLE_ALTERNATION = True    # Lead/lag unit rotation
ENABLE_HYSTERESIS = True     # Per-unit hysteresis bands
LOAD_NOISE_STD = 100.0       # Load noise [J/s]
```
