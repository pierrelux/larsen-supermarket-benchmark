# Bayesian Optimization for PID Tuning

This directory contains optional code for automatic PID parameter tuning using Bayesian Optimization.

## Overview

The main benchmark focuses on comparing MPC controllers against a PID baseline. This extension allows you to tune the PID parameters (Kp, τ_I, DB) automatically using Bayesian Optimization with the Ax platform.

## Installation

```bash
# From the repository root
pip install ax-platform
```

## Usage

```bash
# Run from the repository root
python extras/bo_tuning/tune_pid_with_bo.py
```

## Files

- `tune_pid_with_bo.py` - Main BO tuning script
- `test_bo_interface.py` - Tests for the BO interface
- `manual_pid_comparison.py` - Manual PID parameter comparison
- `BO_TUNING_GUIDE.md` - Detailed guide for BO tuning
- `BO_FILES_SUMMARY.md` - Summary of BO-related files
- `QUICKSTART_BO.md` - Quick start guide
- `CHANGELOG_BO.md` - Changelog for BO features

## Features

- **Automatic parameter search**: Optimizes Kp, τ_I, and DB using Bayesian Optimization
- **Scalar objective**: Combines γ_con (constraint violation), γ_switch (switching rate), and γ_pow (power)
- **Hard constraint penalties**: Prevents unsafe solutions
- **Multi-seed support**: Robust optimization under stochastic load noise
- **Both scenarios**: Works with 2d-2c and 3d-3c configurations

See `BO_TUNING_GUIDE.md` for detailed documentation.

