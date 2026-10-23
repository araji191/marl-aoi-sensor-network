# Decentralized MARL for AoI Optimization in Wireless IoT Networks

This repository contains the source code and simulation environment for the paper: **Decentralized Reinforcement Learning for Age-of-Information Optimization in Energy-Constrained Wireless IoT Networks** by Abiola Raji and Aresh Dadlani (Mount Royal University).

## Overview

Timely status updating is critical in wireless IoT monitoring systems, particularly for applications like wildfire detection where stale sensor readings degrade situational awareness. This project provides a fully decentralized Multi-Agent Reinforcement Learning (MARL) framework to minimize the Age of Information (AoI) in energy-constrained wireless sensor networks.

Each sensor acts as an independent tabular Q-learning agent, adapting its transmission policy based purely on local observations (battery state, transmission status, and capped AoI) without relying on centralized scheduling.

## Repository Structure

* **`MARL_AoI_Simulation.ipynb`**: The interactive Jupyter Notebook. This unified file contains the core simulation classes, training logic, baseline comparisons (Round Robin and Random Access), and plotting functions.

## Requirements

The simulation relies on standard Python libraries. To install the necessary dependencies, run:

```bash
pip install numpy matplotlib
```

## Usage

Open `MARL_AoI_Simulation.ipynb` in Jupyter or VS Code. Adjust the global configuration variables at the top of the notebook (e.g., `NUM_SENSORS`, `NUM_EPISODES`, `ALPHA`) and run all cells to train the agents, evaluate the policies, and generate convergence and average AoI comparison graphs.

## Citation

If you use this code or framework in your research, please cite our paper:

```bibtex
@article{raji2026decentralized,
  title={Decentralized Reinforcement Learning for Age-of-Information Optimization in Energy-Constrained Wireless IoT Networks},
  author={Raji, Abiola and Dadlani, Aresh},
  journal={arXiv preprint},
  year={2026}
}
```