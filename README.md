# Source code repository for Simulation of Complex Systems

#### Authors:
[Daniel Peña Fonseca](https://github.com/dpfonseca)
[Lukas Prader](https://github.com/luprader)

This is the code used for simulations and analyses contained in the project "Characterizing Shortest-Path Ensembles for Brain Network Modeling Using Empirical Constraints".

The project was part of the master course "Simulation Of Complex Systems" at Chalmers University of Technology.

## Project setup and management with uv
https://docs.astral.sh/uv/

### Install uv
```bash
pipx install uv==0.9.5
```

### Sync (install) project dependencies
```bash
uv sync
```
### Run project code
Python scripts can be run using
```bash
uv run filename.py
```
The .ipynb files just need to select the uv .venv as the kernel to run.