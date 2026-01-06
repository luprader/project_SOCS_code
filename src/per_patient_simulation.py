"""Simulates time series and computes correlations for each patient using k-distance matrices for coupling."""

import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm

from simulate_time_series import run_kuramoto


def worker(args):
    """Wrapper needed for multiprocessing.Pool to unpack args."""
    patient_number = args

    k_distance_array = np.load(f"output/k_shortest_paths/{patient_number}.npz")[
        "k_distance_array"
    ]

    patient_ts = (
        pd.read_excel(
            f"NetworkModelling/data/fMRI/{patient_number}.xlsx",
            header=None,
        )
        .to_numpy()
        .T
    )

    patient_fc = np.corrcoef(patient_ts)
    patient_triu = patient_fc[np.triu_indices(patient_fc.shape[0], k=1)]

    k_results = []
    for k_idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 24, 34, 49]:
        adj_mat = 1 / k_distance_array[:, :, k_idx]
        adj_mat[np.isnan(adj_mat)] = 0

        dist_mat = k_distance_array[:, :, k_idx]
        dist_mat[np.isnan(dist_mat)] = 0  # with kdist isnan

        phases = run_kuramoto(
            C=adj_mat,
            distance_matrix=dist_mat,
            dt=0.0001,
            total_time=5,
            coupling_factor=0.2098 * 18,
            noise_factor=1,
            mean_delay=11e-3,
        )

        simulated_ts = np.sin(phases[:, 5000:])  # ignore initial 500ms
        simulated_fc = np.corrcoef(simulated_ts)

        sim_triu = simulated_fc[np.triu_indices(simulated_fc.shape[0], k=1)]

        pearson = sp.stats.pearsonr(sim_triu, patient_triu)[0]

        k_results.append([k_idx + 1, pearson])

    np.save(f"output/simulation_results/{patient_number}.npy", np.array(k_results))
    return patient_number


# Main execution
if __name__ == "__main__":
    start_time = time.time()

    patient_numbers = [
        f.split(".")[0]
        for f in os.listdir("NetworkModelling/data/DTI/")
        if f.endswith(".xlsx")
    ]

    os.makedirs("output/simulation_results/", exist_ok=True)
    already_done = [
        f.split(".")[0]
        for f in os.listdir("output/simulation_results/")
        if f.endswith(".npy")
    ]

    not_calculated = list(set(patient_numbers) - set(already_done))

    k_max = 50
    task_args = not_calculated

    # Run in parallel with a progress bar

    with Pool(6) as p:
        list(tqdm(p.imap(worker, task_args), total=len(task_args)))

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
