import numpy as np
import pandas as pd
from simulate_time_series import run_kuramoto, calculate_bold

import os
import time
from multiprocessing import Pool
import scipy as sp
from tqdm import tqdm

def worker(args):
    """Wrapper needed for multiprocessing.Pool to unpack args."""
    k, coupling_factor, noise_factor = args

    k_distance_average = np.load(f"output/averaged_patient_results.npz")["k_distance_average"]

    dist_mat =  k_distance_average[:, :, k]
    dist_mat[np.isnan(dist_mat)] = 0 # with kdist isnan

    adj_mat = 1 / k_distance_average[:, :, k]
    adj_mat[np.isinf(adj_mat)] = 0

    adj_mat = adj_mat / np.mean(adj_mat) # Normalised to have mean 1

    time_step = 0.1  # 100 ms time resolution

    phases = run_kuramoto(
        C=adj_mat,
        distance_matrix=dist_mat,
        dt=time_step,
        total_time=100,
        noise_factor=noise_factor,
        coupling_factor = coupling_factor,
    )

    activity = np.sin(phases)

    bold = calculate_bold(
        time_series=activity,
        time_step=time_step,
        sample_rate=0.72,
    )

    bold_zscores = sp.stats.zscore(bold, axis=1)

    bold_fc = np.corrcoef(bold_zscores)

    fc_average = np.load(f"output/averaged_patient_results.npz")["fc_average"]

    bold_triu = bold_fc[np.triu_indices(bold_fc.shape[0], k=1)]
    fc_triu = fc_average[np.triu_indices(fc_average.shape[0], k=1)]

    pearson_corr = sp.stats.pearsonr(
        bold_triu,
        fc_triu,
    )[0]

    return pearson_corr

# Main execution
if __name__ == "__main__":
    start_time = time.time()
    # read file names from NetworkModelling/data/DTI/

    patient_numbers = [
        f.split(".")[0]
        for f in os.listdir("NetworkModelling/data/DTI/")
        if f.endswith(".xlsx")
    ]

    if not os.path.exists("output/averaged_patient_results.npz"):
        # compute running average of k-distance arrays and FC matrices
        for patient_idx, patient_number in tqdm(enumerate(patient_numbers), desc="Averaging patient data"):
            k_array = np.load(f"output/k_shortest_paths/{patient_number}.npz")["k_distance_array"]
            k_array[np.isnan(k_array)] = 0  # Set nan to 0 for averaging

            patient_ts_file = pd.ExcelFile(f"NetworkModelling/data/fMRI/{patient_number}.xlsx")
            patient_ts = (
                patient_ts_file.parse(sheet_name="Sheet1", header=None).to_numpy().T
            )

            patient_fc = np.corrcoef(patient_ts)

            if patient_idx == 0:
                k_distance_average = np.zeros_like(k_array)
                fc_average = np.zeros_like(patient_fc)

            k_distance_average += k_array / len(patient_numbers)
            fc_average += patient_fc / len(patient_numbers)

        np.savez_compressed(
            f"output/averaged_patient_results.npz",
            k_distance_average=k_distance_average,
            fc_average=fc_average,
        )

    # Define parameter grids
    k_values = list(np.arange(50, 0, -10))
    coupling_factors = [1, 50, 250, 500, 750]
    noise_factors = [0, 50, 250, 500, 750]
    
    task_args = [(int(k-1), cf, nf) for k in k_values for cf in coupling_factors for nf in noise_factors]

    # Run in parallel with a progress bar
    with Pool(4) as p:
        results = list(tqdm(p.imap(worker, task_args), total=len(task_args), desc="Parameter sweep"))

    # Reshape results into 3D array: (k_values, coupling_factors, noise_factors)
    pearson_array = np.array(results).reshape(len(k_values), len(coupling_factors), len(noise_factors))

    # Save results
    np.savez_compressed(
        f"output/parameter_sweep_results.npz",
        pearson_array=pearson_array,
        k_values=k_values,
        coupling_factors=coupling_factors,
        noise_factors=noise_factors,
    )

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to output/parameter_sweep_results.npz")
    print(f"Pearson array shape: {pearson_array.shape}")