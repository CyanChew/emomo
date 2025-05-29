# HoMeR

[![Paper](https://img.shields.io/badge/Paper-%20%F0%9F%93%84-blue)](TODO)  
[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%90-orange)](TODO)

---

## Overview

**HoMeR** (Hybrid Whole-Body Policies for Mobile Robots) is a hybrid imitation learning framework for mobile manipulation. It combines whole-body control with a hybrid action representation to achieve generalizable and precise robot behavior in both simulation and real-world settings.

---

## Quick Start

Depending on your use case, please follow the appropriate setup and usage instructions:

### 🖥️ Simulation-Only

If you are **only using simulation**, refer to:

📄 [`SIM.md`](SIM.md)

This guide covers:
- Conda setup on macOS and Linux
- Simulated data collection and annotation
- Training and evaluating HoMeR and baselines in simulation

---

### 🤖 Real-World

If you plan to use HoMeR in **real (and optionally simulation)**, refer to:

📄 [`REAL.md`](REAL.md)

This guide covers:
- Hardware and software setup for real-world deployment
- Real-world data collection and annotation
- Training and evaluating HoMeR and baselines in real

---

## Repository Structure

```bash
cfgs/                 # Training config files
envs/                 # Environment setup for sim and real
docker/               # Real-world Docker setup
scripts/              # Training and evaluation scripts
interactive_scripts/  # Data collection, replay, and data annotation tools
dataset_utils/        # Dataset loading and data visualization tools
mj_assets/            # MJCF assets for simulation
sbatch_scripts/       # SLURM scripts to launch training jobs

