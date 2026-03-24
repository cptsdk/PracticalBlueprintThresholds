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


# THRESHOLDS at N=6

THRESHOLDS = {
    'Photon loss':  0.078,
    'Spin depol':   0.0034,
    'EE dist':      0.039,
    'Branching':    0.00174,
    'SE dist':      0.0052,
    'Blinking':     0.056}


# TWO OPERATING TARGET POINTS — exact values from combined_noise_distance_sweep

# noise_vals = (p_spin_depol, p_photon_loss, p_ee_dist, p_branching, p_se_dist)
PARAM_SETS = {
    'Target A': {
        'noise': (0.0034*0.08,
                0.078*0.3, 
                0.039*0.08, 
                0.00174*0.08,
                0.0052*0.08),
        'A': 1,
        'D': 0.00001,
    },
    'Target B': {
        'noise': (0.0034*0.04,
                0.078*0.3, 
                0.039*0.04, 
                0.00174*0.04,
                0.0052*0.04),
        'A': 1,
        'D': 0.00001,
    }}

# Calc Σ p/p_th for display
for name, cfg in PARAM_SETS.items():
    p_spin, p_loss, p_ee, p_br, p_se = cfg['noise']
    D = cfg['D']
    total = (p_spin/0.0034 + p_loss/0.078 + p_ee/0.039 +
             p_br/0.00174 + p_se/0.0052 + D/0.056)
    cfg['sum_label'] = f'{total:.2f}'

# Source index in noise_vals tuple
SOURCE_INDICES = {
    'Photon loss': 1,
    'Spin depol':  0,
    'EE dist':     2,
    'Branching':   3,
    'SE dist':     4}

def make_leave_one_out(cfg):
    """Generate LEAVE_ONE_OUT configs from a param set."""
    ALL_NOISE = cfg['noise']
    ALL_A = cfg['A']
    ALL_D = cfg['D']

    loo = {}
    for name, idx in SOURCE_INDICES.items():
        noise = list(ALL_NOISE)
        noise[idx] = 0.0
        loo[name] = {'noise': tuple(noise), 'A': ALL_A, 'D': ALL_D}
    loo['Blinking'] = {'noise': ALL_NOISE, 'A': ALL_A, 'D': 0}

    return loo


# SIM SETTINGS

L = 3
M_LIST = [6]
NUM_TRIALS = 100000  # Total trials
BATCH_SIZE = 5000  # Max per batch to avoid OOM
NOISE_MECHANISM = 'all_combined'

ROMAN = {1:'i', 2:'ii', 3:'iii', 4:'iv', 5:'v', 6:'vi', 7:'vii', 8:'viii'}

file_name = 'error_budget_data_N6 2.pickle'
with open(os.path.join(saving_folder, file_name) , 'rb') as file:
    all_data = pickle.load(file)

all_results = all_data['results']


# HELPERS

def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.linewidth': 0.8, 'axes.labelsize': 10,
        'axes.titlesize': 11, 'xtick.major.width': 1, 'ytick.major.width': 0.6,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'figure.facecolor': 'white', 'axes.facecolor': 'white'})

def run_one(noise_tuple, L_val, m, A, D):
    data = decoder_successprob_error_vs_loss_list_parallelized_encoded(
        np.array([noise_tuple]), p_fusion_fail=0.5, L=L_val, m=m, A=A, D=D,
        num_loss_trials=NUM_TRIALS, num_ec_runs_per_loss_trial=1,
        noise_mechanism=NOISE_MECHANISM, decoding_weights='None')
    return data[0]

def run_one_batched(noise_tuple, L_val, m, A, D):
    """Run in batches to avoid OOM, accumulate error counts."""
    total_errors = 0
    total_run = 0
    remaining = NUM_TRIALS

    while remaining > 0:
        n = min(BATCH_SIZE, remaining)
        data = decoder_successprob_error_vs_loss_list_parallelized_encoded(
            np.array([noise_tuple]), p_fusion_fail=0.5, L=L_val, m=m, A=A, D=D,
            num_loss_trials=n, num_ec_runs_per_loss_trial=1,
            noise_mechanism=NOISE_MECHANISM, decoding_weights='None')
        total_errors += round(data[0] * n)
        total_run += n
        remaining -= n

    return total_errors / total_run


# RUN

def run_all():
    all_results = {}
    t_total = time.time()

    for set_name, cfg in PARAM_SETS.items():
        ALL_NOISE = cfg['noise']
        ALL_A = cfg['A']
        ALL_D = cfg['D']
        LEAVE_ONE_OUT = make_leave_one_out(cfg)

        results = {}
        for m in M_LIST:
            print(f"\n{'='*60}")
            print(f"  {set_name} (Σ={cfg['sum_label']}), N={m}")
            print(f"{'='*60}")

            p_all = run_one_batched(ALL_NOISE, L, m, ALL_A, ALL_D)
            print(f"  All errors ON: p_L = {p_all:.4e}")

            loo_results = {}
            for name, loo_cfg in LEAVE_ONE_OUT.items():
                p_without = run_one_batched(loo_cfg['noise'], L, m, loo_cfg['A'], loo_cfg['D'])
                delta = p_all - p_without
                loo_results[name] = {'p_without': p_without, 'delta': delta}
                print(f"    Without {name:15s}: p_L = {p_without:.4e}  -> delta = {delta:+.4e}")

            sum_deltas = sum(v['delta'] for v in loo_results.values())
            residual = p_all - sum_deltas
            print(f"  Sum of delta = {sum_deltas:.4e}")
            print(f"  Residual (nonlinear) = {residual:.4e} ({residual/p_all*100:.1f}% of total)")

            results[m] = {
                'p_all': p_all,
                'leave_one_out': loo_results,
                'sum_deltas': sum_deltas,
                'residual': residual}

        all_results[set_name] = results

    print(f"\nTotal time: {time.time() - t_total:.0f}s")
    return all_results

def print_summary(all_results):
    names = list(SOURCE_INDICES.keys()) + ['Blinking']

    for set_name, results in all_results.items():
        for m in sorted(results.keys()):
            res = results[m]
            p_all = res['p_all']
            loo = res['leave_one_out']
            residual = res['residual']

            sum_label = PARAM_SETS[set_name]['sum_label']
            print(f"\n  {set_name} (Σ={sum_label}), N={m}: p_L(all) = {p_all:.4e}")
            header_without = r'p_L(\i)'
            print(f"  {'Source':<15s} {header_without:>12s} {'dp_L':>12s} {'% of total':>10s}")
            print(f"  {'-'*51}")
            for name in names:
                pw = loo[name]['p_without']
                delta = loo[name]['delta']
                pct = delta / p_all * 100 if p_all > 0 else 0
                print(f"  {name:<15s} {pw:>12.4e} {delta:>+12.4e} {pct:>9.1f}%")
            pct_r = residual / p_all * 100 if p_all > 0 else 0
            print(f"  {'Residual':<15s} {'':>12s} {residual:>+12.4e} {pct_r:>9.1f}%")
            print(f"  {'TOTAL':<15s} {'':>12s} {p_all:>12.4e} {'100.0':>9s}%")