"""Simulate activation based on k-shortest paths."""

import os
import time
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

rng = np.random.default_rng()


def simulate_time_series(patient_number: str, time_steps: int, k_used: int) -> np.ndarray:
    """Simulate time series based on k-shortest paths.

    Args:
        patient_number (str): Identifier for the patient.
        time_steps (int): Number of time steps to simulate.
        k_used (int): Number of shortest paths used in the simulation.

    Returns:
        None

    """
    patient_data = np.load(f"output/k_shortest_paths/{patient_number}.npz")
    path_array = patient_data["path_array"]
    distance_array = patient_data["distance_array"][:,:, :k_used]

    # Make the path and distance arrays symmetric
    for i in range(distance_array.shape[0]):
        for j in range(i):
            if i == j:
                continue

            path_array[j, i] = path_array[i, j]
            distance_array[j, i] = distance_array[i, j]


    sums = distance_array.sum(axis=2, keepdims=True)
    sums_safe = np.where(sums == 0, 1.0, sums)

    p_path_selection = distance_array / sums_safe


    active_nodes = np.arange(path_array.shape[0])
    target_nodes = rng.integers(0, path_array.shape[0], size=active_nodes.shape[0])

    time_series = np.zeros((path_array.shape[0], time_steps), dtype=np.int32)


    for t in range(time_steps):
        next_step = time_series[:, t]
        for start, target in zip(active_nodes, target_nodes):
            
            k_shortest_paths = path_array[start, target]
            path_probabilities = p_path_selection[start, target]
            path_choice = rng.choice(
                np.arange(k_shortest_paths.shape[0]),
                p=path_probabilities,
            )

            chosen_path = k_shortest_paths[path_choice]

            traversed_nodes = [target]
            while chosen_path[traversed_nodes[-1]] != start:
                traversed_nodes.append(int(chosen_path[traversed_nodes[-1]]))

            next_step[traversed_nodes] = 1

    return time_series
    # np.save(
    #     f"output/simulated_time_series/{patient_number}.npy",
    #     time_series,
    # )


test = simulate_time_series("103414", 1000, 5)

# def worker(args):
#     """Wrapper needed for multiprocessing.Pool to unpack args."""
#     patient_number, time_steps = args

#     simulate_time_series(path_array, distance_array, time_steps)
#     return patient_number


# # Main execution
# if __name__ == "__main__":
#     start_time = time.time()
#     # read file names from NetworkModelling/data/DTI/

#     patient_numbers = [
#         f.split(".")[0]
#         for f in os.listdir("NetworkModelling/data/DTI/")
#         if f.endswith(".xlsx")
#     ]

#     os.makedirs("output/simulated_time_series/", exist_ok=True)
#     already_done = [
#         f.split(".")[0]
#         for f in os.listdir("output/simulated_time_series/")
#         if f.endswith(".npy")
#     ]

#     not_calculated = list(set(patient_numbers) - set(already_done))

#     task_args = [(p, k_max) for p in not_calculated]

#     # Run in parallel with a progress bar

#     with Pool(6) as p:
#         list(tqdm(p.imap(worker, task_args), total=len(task_args)))

#     end_time = time.time()
#     print(f"Elapsed time: {end_time - start_time:.2f} seconds")
