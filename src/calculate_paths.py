"""Calculate k-shortest paths between ROIs based on DTI data."""

import os
import time

import numpy as np
import pandas as pd
import scipy as sp


def calculate_k_shortest_paths(patient_number: str, k: int) -> None:
    """Calculate k-shortest paths between nodes with connectivity matrix.

    Args:
        patient_number (str): Identifier for the patient data file.
        k (int): Number of shortest paths to calculate.

    Returns:
        None

    """
    # Load the connectivity matrix from an Excel file
    patient_file = pd.ExcelFile(f"NetworkModelling/data/DTI/{patient_number}.xlsx")
    patient_df = patient_file.parse(sheet_name="Sheet1", header=None)

    path_array = np.zeros(
        (patient_df.shape[0], patient_df.shape[0], k, patient_df.shape[0] + 1),
    )

    for i in range(patient_df.shape[0]):
        for j in range(i):
            if i == j:
                continue

            k_shortest_result = sp.sparse.csgraph.yen(
                patient_df,
                i,
                j,
                k,
                directed=False,
                return_predecessors=True,
            )
            combined_data = np.hstack(
                (k_shortest_result[0].reshape(-1, 1), k_shortest_result[1]),
            )
            if combined_data.size > 0:
                path_array[i, j] = combined_data

    if not os.path.exists("output/k_shortest_paths/"):
        os.makedirs("output/k_shortest_paths/")
    np.save(f"output/k_shortest_paths/{patient_number}.npy", path_array)


# Main execution
start_time = time.time()
# read file names from NetworkModelling/data/DTI/


patient_numbers = [
    f.split(".")[0]
    for f in os.listdir("NetworkModelling/data/DTI/")
    if f.endswith(".xlsx")
]

for patient_number in patient_numbers[:3]:
    calculate_k_shortest_paths(patient_number, 20)

end_time = time.time()
print(f"Elapsed time: {end_time - start_time} seconds")
