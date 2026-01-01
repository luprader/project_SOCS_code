import os

import numpy as np
import pandas as pd
from tqdm import tqdm

patient_numbers = [
    f.split(".")[0]
    for f in os.listdir("NetworkModelling/data/DTI/")
    if f.endswith(".xlsx")
]

n_patients = len(patient_numbers)

average_fc = np.zeros((246, 246))
average_kdistance = np.zeros((246, 246, 50))
average_sc = np.zeros_like(average_fc)

for patient_number in tqdm(patient_numbers):
    k_distances =np.load(f"output/k_shortest_paths/{patient_number}.npz")[
        "k_distance_array"
    ]
    k_distances[np.isnan(k_distances)] = 0
    average_kdistance += k_distances


    patient_ts = (
        pd.read_excel(
            f"NetworkModelling/data/fMRI/{patient_number}.xlsx",
            header=None,
        )
        .to_numpy()
        .T
    )
    average_fc += np.corrcoef(patient_ts)

    average_sc += (
        pd.read_excel(
            f"NetworkModelling/data/DTI/{patient_number}.xlsx",
            header=None,
        )
        .to_numpy()
        .T
    )


np.savez_compressed(
    "output/averaged_patient_results.npz",
    fc_average=average_fc / n_patients,
    k_distance_average=average_kdistance / n_patients,
    sc_average=average_sc / n_patients,
)
