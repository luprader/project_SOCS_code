import time

import numpy as np
import scipy as sp

from simulate_time_series import run_kuramoto


def evaluate_simulation_avg(k_list, n_rep, scale_factor):
    k_distance_array = np.load("output/averaged_patient_results.npz")[
        "k_distance_average"
    ]

    patient_fc = np.load("output/averaged_patient_results.npz")["fc_average"]

    pearson_list = []
    for k_idx in k_list:

        adj_mat = 1 / k_distance_array[:, :, k_idx]
        adj_mat[np.isnan(adj_mat)] = 0

        dist_mat = k_distance_array[:, :, k_idx]
        dist_mat[np.isnan(dist_mat)] = 0  # with kdist isnan

        corr_val = 0
        for rep in range(n_rep):
            phases = run_kuramoto(
                C=adj_mat,
                distance_matrix=dist_mat,
                dt=0.0001,
                total_time=2,
                coupling_factor=scale_factor * 18,
                noise_factor= 1,
                mean_delay=11e-3,
                # seed=0,
            )

            simulated_ts = np.sin(phases[:, 5000:])  # ignore initial 500ms

            simulated_fc = np.corrcoef(simulated_ts)

            sim_triu = simulated_fc[np.triu_indices(simulated_fc.shape[0], k=1)]
            patient_triu = patient_fc[np.triu_indices(patient_fc.shape[0], k=1)]

            pearson = sp.stats.pearsonr(sim_triu, patient_triu)[0]

            corr_val += pearson / n_rep

        pearson_list.append(float(corr_val))

    print(scale_factor, pearson_list)
    return np.mean(pearson_list)


# Main execution
if __name__ == "__main__":
    start_time = time.time()

    klist = [0, 4, 9, 29]
    n_rep = 3

    result = sp.optimize.minimize_scalar(lambda scale_factor: -evaluate_simulation_avg(klist, n_rep, scale_factor), bounds=(0.05, 1), options={"maxiter":10}, method='bounded')

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    print(result)
