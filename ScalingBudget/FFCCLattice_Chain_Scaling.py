import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pickle

sys.path.insert(0, '.')
sys.path.insert(0, '../')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
cwd = os.getcwd()
saving_folder = os.path.join(cwd, "SavingFolder")

from Decoding.Decoder import decoder_successprob_error_vs_loss_list_parallelized_encoded

"""
Combined-noise distance sweep: ALL errors simultaneously (including blinking)

Runs the decoder with every error source active, sweeps code distance L,
fits log10(p_L) vs d, and extrapolates to find the distance needed
for target logical error rates.
"""


# PARAMETER SETS
# noise_vals = (p_spin_depol, p_photon_loss, p_ee_dist, p_branching, p_se_dist)

PARAM_SETS = {    
    # Thresholds at N=6
    # 'Spin depol':   0.0034,
    # 'Photon loss':  0.078,
    # 'EE dist':      0.039,
    # 'Branching':    0.00174,
    # 'SE dist':      0.0052,
    # 'Blinking':     0.056,

    'Nearer-term': { 
        'noise': (0.0034*0.08,
                0.078*0.3, 
                0.039*0.08, 
                0.00174*0.08,
                0.0052*0.08),
        'A': 1,
        'D': 0.00001, 
        'color': '#440154',
        'marker': 'v',
        'label': f'8% of error threshold, 30% of loss threshold'},
    
    'Target': { 
        'noise': (0.0034*0.04,
                0.078*0.3, 
                0.039*0.04, 
                0.00174*0.04,
                0.0052*0.04),
        'A': 1,
        'D': 0.00001, 
        'color': '#31688e',
        'marker': 'D',
        'label': f'4% of error threshold, 30% of loss threshold'}}

# Code parameters
M = 6
L_LIST = [3, 4, 5]
P_FUSION_FAIL = 0.5
NUM_TRIALS = 10000
NOISE_MECHANISM = 'RUS_all_errors'

# Targets for extrapolation
TARGET_6  = 1e-6
TARGET_9  = 1e-9
TARGET_12 = 1e-12


#  RUN

def run_one(noise_tuple, L, m, A, D, n_trials):
    data = decoder_successprob_error_vs_loss_list_parallelized_encoded(
        np.array([noise_tuple]), p_fusion_fail=P_FUSION_FAIL, L=L, m=m,
        A=A, D=D, num_loss_trials=n_trials, num_ec_runs_per_loss_trial=1,
        noise_mechanism=NOISE_MECHANISM, decoding_weights='None')
    return data[0]

def run_one_batched(noise_tuple, L, m, A, D, total_trials, batch_size=2000):
    """Run in batches to avoid OOM, accumulate error counts."""

    n_batches = (total_trials + batch_size - 1) // batch_size
    total_errors = 0
    total_run = 0

    for b in range(n_batches):
        n = min(batch_size, total_trials - total_run)
        p_L = run_one(noise_tuple, L, m, A, D, n)
        total_errors += round(p_L * n)
        total_run += n

    return total_errors / total_run

def run_sweep(name, cfg):
    noise = cfg['noise']
    A, D = cfg['A'], cfg['D']
    p_spin, p_loss, p_ee, p_br, p_se = noise

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  p_spin={p_spin}, p_loss={p_loss}, p_ee={p_ee}, p_br={p_br}, p_se={p_se}")
    print(f"  A={A}, D={D}, m={M}")
    print(f"{'='*70}")

    # Print p/p_th budget
    thresholds = [0.0034, 0.078, 0.039, 0.00174, 0.0052, 0.056]
    values = [p_spin, p_loss, p_ee, p_br, p_se, D]
    labels = ['spin', 'loss', 'ee_dist', 'branch', 'se_dist', 'blink']
    total_ratio = 0
    for lab, val, th in zip(labels, values, thresholds):
        r = val / th
        total_ratio += r
        print(f"    {lab:>10s}: p/p_th = {r:.3f}")
    print(f"    {'SUM':>10s}: {total_ratio:.3f}  {'(< 1 OK)' if total_ratio < 1 else '(>= 1 WARNING!)'}")

    results = {}
    t0 = time.time()
    for L in L_LIST:
        t1 = time.time()
        # p_L = run_one_batched(noise, L, M, A, D, total_trials=5000, batch_size=10000)
        p_L = run_one_batched(noise, L, M, A, D, total_trials=10000, batch_size=5000)
        dt = time.time() - t1
        results[L] = p_L
        print(f"    L={L}: p_L = {p_L:.4e}  ({dt:.0f}s)")
    print(f"  Subtotal: {time.time() - t0:.0f}s")
    return results

def fit_and_extrapolate(results, label):
    Ls = np.array([L for L in sorted(results.keys()) if results[L] > 0])
    pLs = np.array([results[L] for L in Ls])

    if len(Ls) < 2:
        print(f"  {label}: <2 nonzero points, cannot fit")
        return None, None, None, None, None

    slope, intercept, r_val, _, _ = linregress(Ls, np.log10(pLs))
    d_6  = (np.log10(TARGET_6)  - intercept) / slope if slope < 0 else float('inf')
    d_9  = (np.log10(TARGET_9)  - intercept) / slope if slope < 0 else float('inf')
    d_12 = (np.log10(TARGET_12) - intercept) / slope if slope < 0 else float('inf')

    print(f"\n  {label} fit: slope = {slope:.4f}, intercept = {intercept:.4f}, R² = {r_val**2:.4f}")
    print(f"    L(1e-6)  ~ {d_6:.1f}")
    print(f"    L(1e-9)  ~ {d_9:.1f}")
    print(f"    L(1e-12) ~ {d_12:.1f}")

    return slope, intercept, d_6, d_9, d_12