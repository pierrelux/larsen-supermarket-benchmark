# ECC'07 Supermarket Refrigeration Benchmark

Implementation of the ECC'07 supermarket refrigeration benchmark from Larsen et al. Features configurable traditional control with advanced diagnostics and per-unit switch counting.

Based on Larsen paper with unspecified implementation details added. Includes capacity-agnostic quantizer, numerical stability guards, and baseline alignment options.

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

## Results

Automatic comparison table shows performance vs. paper baseline:
- Clean baseline: 87-98% lower γ_switch than paper
- Alignment features: Increase switching closer to paper values
- All metrics: γ_con, γ_switch, γ_pow for day/night periods

## Features

- Per-unit compressor switch counting
- VFD control with configurable minimum hold
- Lead/lag unit alternation
- Hysteresis bands for realistic rack behavior
- Load noise injection for valve synchronization
- Numerical stability guards
- Comprehensive diagnostics
- Reproducible results with random seeding
