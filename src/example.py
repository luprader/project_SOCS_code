# ...existing code...
"""
Python conversion of the provided MATLAB routine to generate example functional data
using modular Watts-Strogatz graphs. Produces three groups of 10 subjects each,
writes per-subject time series as .xlsx files and writes a vois (Subject ID, Age, Sex)
file per group.

Usage (from project root or the file's directory):
    python3 example.py /path/to/output_dir 42
If random_seed is omitted, a default RNG is used.
"""
from pathlib import Path
import numpy as np
import pandas as pd

def watts_strogatz_adj(N, K, beta, rng):
    """
    Construct adjacency matrix following the MATLAB test WattsStrogatz implementation.
    N: number of nodes
    K: number of nearest neighbors (on one side)
    beta: rewiring probability (0..1)
    rng: numpy Generator
    Returns NxN adjacency (0/1) symmetric, no self-loops.
    """
    if N <= 0:
        return np.zeros((0,0), dtype=int)
    # ensure K valid
    K = int(max(1, min(K, N-1)))
    # create ring lattice edges (directed pairs s->t)
    s = np.repeat(np.arange(N), K)
    t = (np.repeat(np.arange(N), K) + np.tile(np.arange(1, K+1), N)) % N
    # reshape to (N, K)
    s2 = s.reshape(N, K)
    t2 = t.reshape(N, K)
    # rewire as in MATLAB implementation
    for source in range(N):
        switch_edge = rng.random(K) < beta  # boolean mask
        if not np.any(switch_edge):
            continue
        # available new targets
        new_targets = rng.random(N)
        # exclude source itself
        new_targets[source] = -1.0
        # exclude nodes that would create parallel edges (sources whose target == source)
        mask = (t2 == source)
        if mask.any():
            new_targets[s2[mask]] = -1.0
        # exclude current non-switched targets
        new_targets[t2[source, ~switch_edge]] = -1.0
        # choose top candidates
        inds = np.argsort(new_targets)[::-1]  # descending
        k_needed = int(switch_edge.sum())
        chosen = inds[:k_needed]
        t2[source, switch_edge] = chosen
    # build adjacency
    adj = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in t2[i]:
            adj[i, j] = 1
            adj[j, i] = 1  # undirected
    np.fill_diagonal(adj, 0)  # ensure no self-loops initially
    return adj

def create_data(data_dir=None, random_seed=None):
    # parameters
    if data_dir is None:
        data_dir = Path.cwd() / 'Example_data_NN_CLA_FUN_XLS'
    else:
        data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # RNG
    rng = np.random.default_rng(None if random_seed in (None, 'default') else int(random_seed))

    sex_options = ['Female', 'Male']

    N_nodes = 90
    N_tslength = 200
    N_groups = 10  # per group

    subject_counter = 1
    for group_idx in range(1, 4):  # 3 groups
        gr_name = f'FUN_Group_{group_idx}_XLS'
        gr_dir = data_dir / gr_name
        gr_dir.mkdir(exist_ok=True)
        vois_rows = []
        # loop subjects
        start_id = (group_idx - 1) * N_groups + 1
        end_id = group_idx * N_groups
        for i_gr in range(start_id, end_id + 1):
            sub_id = f'SubjectFUN_{i_gr}'
            # modules
            N_module = rng.integers(1, 9)  # 1..8 inclusive
            K_temp = np.arange(1, 1 + N_module)
            K = rng.permutation(K_temp)
            beta = rng.random(N_module)

            # partition nodes
            L_indice = N_nodes // N_module
            indices = []
            for j in range(N_module):
                if j != N_module - 1:
                    start = j * L_indice
                    end = start + L_indice
                else:
                    start = j * L_indice
                    end = N_nodes
                indices.append(np.arange(start, end))

            A_full = np.zeros((N_nodes, N_nodes), dtype=float)
            for i_mod in range(N_module):
                idx = indices[i_mod]
                n_mod = len(idx)
                k_val = int(K[i_mod])
                # cap k to valid range
                k_val = max(1, min(k_val, n_mod - 1)) if n_mod > 1 else 1
                adj_block = watts_strogatz_adj(n_mod, k_val, float(beta[i_mod]), rng)
                A_full[np.ix_(idx, idx)] = adj_block

            # set diagonal to 1 (as in MATLAB)
            np.fill_diagonal(A_full, 1.0)

            # make positive-definite covariance
            cov = A_full @ A_full.T
            # numerical stabilizer
            cov += 1e-8 * np.eye(N_nodes)

            mu_gr = np.ones(N_nodes)
            # draw multivariate normal samples (N_tslength x N_nodes)
            try:
                R = rng.multivariate_normal(mu_gr, cov, size=N_tslength)
            except Exception:
                # fall back to using cholesky
                L = np.linalg.cholesky(cov)
                Z = rng.standard_normal(size=(N_tslength, N_nodes))
                R = Z @ L.T + mu_gr

            # column-wise z-score (match MATLAB behavior)
            col_mean = R.mean(axis=0)
            col_std = R.std(axis=0)
            col_std[col_std == 0] = 1.0
            Rz = (R - col_mean) / col_std

            # save subject timeseries to xlsx without headers/index
            df = pd.DataFrame(Rz)
            df.to_excel(gr_dir / f'{sub_id}.xlsx', index=False, header=False)

            # variables of interest row
            age = int(rng.integers(1, 91))  # 1..90 like randi(90)
            sex = rng.choice(sex_options)
            vois_rows.append({'Subject ID': sub_id, 'Age': age, 'Sex': sex})

            subject_counter += 1

        # write vois file for group
        vois_df = pd.DataFrame(vois_rows, columns=['Subject ID', 'Age', 'Sex'])
        vois_df.to_excel(data_dir / f'{gr_name}.vois.xlsx', index=False)

if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else None
    seed = sys.argv[2] if len(sys.argv) > 2 else None
    create_data(out, seed)
# ...existing code...