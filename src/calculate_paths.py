"""Calculate k-shortest paths between ROIs based on DTI data."""

import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm


def reconstruct_path(predecessors: np.ndarray, start: int, end: int) -> list[int]:
    """Reconstruct path from predecessors array.

    Args:
        predecessors (np.ndarray): Array of predecessors from shortest path algorithm.
        start (int): Starting node index.
        end (int): Ending node index.

    Returns:
        list[int]: List of node indices representing the path from start to end.

    """
    path = []
    current = end
    while current != start:
        path.append(int(current))
        current = predecessors[current]
        if current == -9999:  # No path exists
            return []
    path.append(start)
    path.reverse()
    return path


def calculate_k_shortest_distances(patient_number: str, k: int) -> np.ndarray:
    """Calculate k-shortest distances between nodes with connectivity matrix.

    Args:
        patient_number (str): Identifier for the patient data file.
        k (int): Number of shortest paths to calculate.

    Returns:
        np.ndarray: 3D array of shape (num_nodes, num_nodes, k) containing k-shortest distances.

    """
    # Load the connectivity matrix from an Excel file
    patient_file = pd.ExcelFile(f"NetworkModelling/data/DTI/{patient_number}.xlsx")
    patient_array = patient_file.parse(sheet_name="Sheet1", header=None).to_numpy(
        dtype=float,
    )

    n = patient_array.shape[0]
    # result: (n, n, k)
    k_distance_array = np.full((n, n, k), np.nan, dtype=np.float64)

    # precompute row sums for normalisation (outgoing sums)
    weight_rowsums = np.sum(patient_array, axis=1)  # axis=1: sum across columns

    # compute distances once and treat zeros as no-edge (inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        distance_array = -np.log(patient_array)
    zero_mask = patient_array == 0
    distance_array[zero_mask] = np.inf

    # iterate strict lower triangle (i>j) — avoids double work
    for i in range(n):
        for j in range(i):
            # Calculate k-shortest paths and predecessor arrays
            k_path_lengths, k_paths = sp.sparse.csgraph.yen(
                distance_array,
                i,
                j,
                k,
                directed=False,
                return_predecessors=True,
            )


            # If no paths found for this pair, leave nan
            if len(k_path_lengths) == 0:
                continue

            # Normalize shapes: k_paths may be 1D or 2D
            k_path_lengths = np.asarray(k_path_lengths, dtype=float)
            k_paths = np.asarray(k_paths)
            if k_paths.ndim == 1:
                k_paths = k_paths.reshape(1, -1)

            num_found = k_paths.shape[0]
            k_found = min(num_found, k_path_lengths.shape[0])

            # Compute per-path unnormalised scores (edge contributions)
            per_path_scores = np.zeros(k_found, dtype=np.float64)
            for p in range(k_found):
                pred = k_paths[p]
                path = reconstruct_path(pred, i, j)  # use your reconstruct_path
                if not path or len(path) < 2:
                    per_path_scores[p] = 0.0
                    continue

                # Vectorised per-edge extraction
                u = np.array(path[:-1], dtype=np.int32)
                v = np.array(path[1:], dtype=np.int32)
                edge_weights = patient_array[u, v]  # vectorised lookup
                denom = weight_rowsums[u]  # outgoing sum for each u

                # Avoid division-by-zero
                valid = denom > 0.0
                if not valid.any():
                    per_path_scores[p] = 0.0
                else:
                    contrib = np.zeros_like(edge_weights, dtype=np.float64)
                    contrib[valid] = edge_weights[valid] / denom[valid]
                    per_path_scores[p] = contrib.sum()

            # Build cumulative probabilities and compute k-shortest distances for all k_used
            if k_found == 0:
                continue

            prob_cumsum = np.cumsum(per_path_scores)
            weighted_len_prefix = np.cumsum(per_path_scores * k_path_lengths[:k_found])

            k_shortest_distances = np.full(k, np.nan, dtype=np.float64)
            for kk in range(1, k_found + 1):
                s = prob_cumsum[kk - 1]
                if s > 0.0:
                    k_shortest_distances[kk - 1] = weighted_len_prefix[kk - 1] / s
                else:
                    # fallback: if all zero contributions, use simple mean of lengths
                    k_shortest_distances[kk - 1] = float(np.mean(k_path_lengths[:kk]))

            # assign for (i,j); mirror later in bulk
            k_distance_array[i, j, :] = k_shortest_distances

    # Mirror strict lower triangle into upper triangle in one vectorised copy
    tril_i, tril_j = np.tril_indices(n, k=-1)
    k_distance_array[tril_j, tril_i, :] = k_distance_array[tril_i, tril_j, :]

    return k_distance_array


def worker(args):
    """Wrapper needed for multiprocessing.Pool to unpack args."""
    patient_number, k = args
    k_array = calculate_k_shortest_distances(patient_number, k)

    np.savez_compressed(
        f"output/k_shortest_paths/{patient_number}.npz",
        k_distance_array=k_array,
    )
    return patient_number


# Main execution
if __name__ == "__main__":
    start_time = time.time()
    # read file names from NetworkModelling/data/DTI/

    patient_numbers = [
        f.split(".")[0]
        for f in os.listdir("NetworkModelling/data/DTI/")
        if f.endswith(".xlsx")
    ]

    os.makedirs("output/k_shortest_paths/", exist_ok=True)
    already_done = [
        f.split(".")[0]
        for f in os.listdir("output/k_shortest_paths/")
        if f.endswith(".npz")
    ]

    not_calculated = list(set(patient_numbers) - set(already_done))

    k_max = 5
    task_args = [(p, k_max) for p in not_calculated]

    # Run in parallel with a progress bar

    with Pool(4) as p:
        list(tqdm(p.imap(worker, task_args), total=len(task_args)))

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
