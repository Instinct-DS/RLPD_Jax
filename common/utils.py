# Utils files to have the helper functions #

import numpy as np
import pickle
from collections import deque
from tqdm import tqdm
from scipy.interpolate import BarycentricInterpolator, CubicSpline

# Nodes Used in OCP #
def lglnodes(N):
    N1 = N + 1
    x = np.cos(np.pi * np.arange(N1) / N)
    P = np.zeros((N1, N1))
    xold = 2 * np.ones_like(x)
    while np.max(np.abs(x - xold)) > np.finfo(float).eps:
        xold = x.copy()
        P[:, 0] = 1
        P[:, 1] = x
        for k in range(2, N1):
            P[:, k] = ((2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k
        x = xold - (x * P[:, N] - P[:, N - 1]) / (N1 * P[:, N])
    w = 2 / (N * N1 * P[:, N]**2)
    return np.flip(x), np.flip(w)    

# CGL nodes #
def findCGL(N):
    """
    Generate Chebyshev–Gauss–Lobatto (CGL) nodes in [-1, 1]
    ordered from -1 to 1, along with the differentiation matrix.
    """
    if N < 2:
        raise ValueError("N must be >= 2")

    p = np.arange(N)
    nds = np.cos(p * np.pi / (N - 1))  # standard CGL nodes (1 → -1)

    D = np.zeros((N, N))
    cbar = np.ones(N)
    cbar[0] = 2
    cbar[-1] = 2

    D[0, 0] = (2 * (N - 1)**2 + 1) / 6
    D[-1, -1] = -D[0, 0]

    for i in range(1, N - 1):
        D[i, i] = -0.5 * nds[i] / (1 - nds[i]**2)

    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (
                    cbar[i] * (-1)**(i + j)
                    / (cbar[j] * (nds[i] - nds[j]))
                )

    # ---- reorder nodes and differentiation matrix ----
    nds = nds[::-1]
    D = D[::-1, ::-1]

    return nds, D

# Get Interpolator #
def getInterpolators(result, nds):
    n = len(nds)
    Z = result
    states = np.reshape(Z[:6*n], shape=(n,6), order='F')
    ctrl = np.reshape(Z[6*n:8*n], shape=(n,2), order="F")
    tf = Z[-1]

    x_i = states[:, 0]
    z_i = states[:, 1]
    u_i = states[:, 2]
    w_i = states[:, 3]
    omg_i = states[:, 4]
    lmbi = states[:, 5]
    tht0 = ctrl[:, 0]
    thtP = ctrl[:, 1]

    x_int = BarycentricInterpolator(nds, x_i)
    z_int = BarycentricInterpolator(nds, z_i)
    u_int = BarycentricInterpolator(nds, u_i)
    w_int = BarycentricInterpolator(nds, w_i)
    omg_int = BarycentricInterpolator(nds, omg_i)
    lmbi_int = BarycentricInterpolator(nds, lmbi)
    tht0_int = BarycentricInterpolator(nds, tht0)
    thtP_int = BarycentricInterpolator(nds, thtP)

    return x_int, z_int, u_int, w_int, omg_int, lmbi_int, tht0_int, thtP_int


# Load expert trajectories into replay buffer #
def load_demo_trajectories(replay_buffer, demo_file, demo_env, nds, gamma=0.999):
    nds, _ = findCGL(nds)
    demo_file = demo_file
    with open(demo_file, "rb") as file:
        TRAJ = pickle.load(file)

    count_land, count_crash = 0, 0 
    test_env = demo_env

    # Loading the trajectories #
    for key in tqdm(TRAJ.keys(), desc="Loading Trajectories", unit="trj"):
        tr_ = TRAJ[key]
        # if tr_["height"] not in np.arange(start=300, stop=405, step=10):
        #     continue
        # if tr_["velocity"] not in np.arange(start=0, stop=21, step=5):
        #     continue

        result = tr_["solution_vector"]

        x_int, z_int, u_int, w_int, omg_int, lmbi_int, tht0_int, thtP_int = getInterpolators(result, nds)
        tf = result[-1]

        success = False
        for eps in np.linspace(0,1,21):
            x_int, z_int, u_int, w_int, omg_int, lmbi_int, tht0_int, thtP_int = getInterpolators(result, nds)
            init = (x_int(-1), z_int(-1)+eps, u_int(-1), w_int(-1), omg_int(-1), tht0_int(-1), thtP_int(-1), lmbi_int(-1))
            action_int = (tht0_int, thtP_int)
            sim_traj = test_env._similate_trajectory(init, action_int, tf)

            if sim_traj[-1][3] and sim_traj[-1][-1]["constraints"][0][2]:
                success = True
                break

        if success:
            count_land += 1
            for i in range(len(sim_traj)):
                experience = sim_traj[i]
                state, action, reward, termination, truncation, next_state, info = experience
                for buffer_i in replay_buffer:
                    buffer_i.add(state, action, reward, termination, truncation, next_state)
        else:
            count_crash += 1
            # tqdm.write(f"Failed : {key}")

    print(f"Loaded all successful trajectories... No of Sucess : {count_land}; No of Crash : {count_crash}")
    return replay_buffer

from joblib import Parallel, delayed, cpu_count
from typing import List, Dict, Any, Tuple
import warnings

def _process_single_trajectory(tr_key: str, tr_data: Dict, demo_env, nds, 
                               gamma: float = 0.999, buffer_lock=None):
    """
    Process a single trajectory for parallel execution
    """
    # Recreate the environment for this thread (important!)
    thread_env = demo_env.__class__()  # Create new instance
    thread_env.__dict__.update(demo_env.__dict__)  # Copy state
    
    result = tr_data["solution_vector"]
    x_int, z_int, u_int, w_int, omg_int, lmbi_int, tht0_int, thtP_int = getInterpolators(result, nds)
    tf = result[-1]
    
    # success = False
    experiences = []
    
    # # Try different epsilon values
    # for eps in np.linspace(0, 1, 21):
    #     x_int, z_int, u_int, w_int, omg_int, lmbi_int, tht0_int, thtP_int = getInterpolators(result, nds)
    #     init = (x_int(-1), z_int(-1)+eps, u_int(-1), w_int(-1), omg_int(-1), 
    #             tht0_int(-1), thtP_int(-1), lmbi_int(-1))
    #     action_int = (tht0_int, thtP_int)
        
    #     try:
    #         sim_traj = thread_env._similate_trajectory(init, action_int, tf)
            
    #         if sim_traj[-1][3] and sim_traj[-1][-1]["constraints"][0][2]:
    #             success = True
    #             # Collect all experiences
    #             for i in range(len(sim_traj)):
    #                 experience = sim_traj[i]
    #                 state, action, reward, termination, truncation, next_state, info = experience
    #                 experiences.append((state, action, reward, termination, truncation, next_state))
    #             break
    #     except Exception as e:
    #         warnings.warn(f"Error processing trajectory {tr_key}: {e}")
    #         continue
    
    success = True
    x_int, z_int, u_int, w_int, omg_int, lmbi_int, tht0_int, thtP_int = getInterpolators(result, nds)
    init = (x_int(-1), z_int(-1), u_int(-1), w_int(-1), omg_int(-1), tht0_int(-1), thtP_int(-1), lmbi_int(-1))
    action_int = (tht0_int, thtP_int)
        
    try:
        sim_traj = thread_env._similate_trajectory(init, action_int, tf)
        # Collect all experiences
        for i in range(len(sim_traj)):
            experience = sim_traj[i]
            state, action, reward, termination, truncation, next_state, info = experience
            experiences.append((state, action, reward, termination, truncation, next_state))
    except Exception as e:
        warnings.warn(f"Error processing trajectory {tr_key}: {e}")

    return success, experiences, tr_key

def load_demo_trajectories_parallel(replay_buffer, demo_file, demo_env, nds, nds_name="CGL",
                                   gamma: float = 0.999, n_jobs: int = -1, n_load=-1):
    """
    Parallel version of load_demo_trajectories using joblib
    
    Parameters:
    -----------
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available CPUs.
    """
    # Process nds once
    if nds_name in ("cgl", "CGL"): 
        nds, _ = findCGL(nds)
    else:
        nds, _ = lglnodes(nds-1)
    
    # Load trajectories
    with open(demo_file, "rb") as file:
        TRAJ = pickle.load(file)
    
    if n_jobs == -1:
        n_jobs = cpu_count() - 2
    print(f"Using {n_jobs} parallel workers")
    
    # Prepare arguments for parallel processing
    tr_keys = list(TRAJ.keys())
    if n_load != -1:
        ind = np.random.permutation(len(tr_keys))[:n_load]
        tr_keys = [tr_keys[i] for i in ind]
    tr_datas = [TRAJ[key] for key in tr_keys]
    
    # Process trajectories in parallel with progress bar
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_single_trajectory)(tr_key, tr_data, demo_env, nds, gamma)
        for tr_key, tr_data in zip(tr_keys, tr_datas))
    
    # Aggregate results
    count_land, count_crash = 0, 0
    
    for success, experiences, tr_key in tqdm(results, desc="Adding to buffers"):
        if success:
            count_land += 1       
            # Add experiences to replay buffers (sequential part)
            for state, action, reward, termination, truncation, next_state in experiences:
                for buffer_i in replay_buffer:
                    buffer_i.add(state, action, reward, termination, truncation, next_state)
        else:
            count_crash += 1
    
    print(f"Loaded all successful trajectories...")
    print(f"Success: {count_land}, Crashes: {count_crash}")
    
    return replay_buffer