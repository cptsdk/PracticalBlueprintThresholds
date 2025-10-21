import numpy as np
import matplotlib.pyplot as plt
import os
import ctypes
import numba
from numba import njit
import multiprocessing
from itertools import repeat
from pymatching import Matching
from math import comb
from collections import Counter
import sys
sys.path.insert(0, '../')

cwd = os.getcwd()
saving_folder = os.path.join(cwd, "SavingFolder")

from Lattice.FFCCLattice_Chain import FFCCLattice_chain
from linear_algebra_inZ2_new import LossDecoder_GaussElimin_trackqbts_noorderedlost
from NoiseSampling_funcs import multiedge_errorprob_uncorrelated, propagate_errors_multiedge, multiedge_errorprob_uncorrelated_precharnoise
from misc_functions import merge_multiedges_in_Hmat_faster, get_multiedge_errorprob, get_Hmat_weights, \
    get_paired_qbt_list, fusion_error_merged

from rsg_error_sampling import encoded_chain_sampler  # original without time-bin


#################### Functions for full parallelized decoder of errors & losses for FFCC lattice ####################

def decoder_single_run_lossandEC_encoded( qubit_errors, loss_mask, m, num_fusions, qbts_in_fusions, num_qubits_res_state, fusions_primal_isZZ, 
                                         fusions_layer_order, fusion_for_qubits, H_withlogop_primal, fus_syndr_mat_primal, H_withlogop_dual, 
                                         fus_syndr_mat_dual, p_loss, p_fusion_success_biased, p_fusion_success_unbiased, p_err, p_err_biased, 
                                         p_err_unbiased, dist_err, num_physical_qbts, qbts_in_resource_states, A, D, num_ec_runs_per_loss_trial=1, 
                                         noise_mechanism='RUS', decoding_weights='None', fusion_is_physical=True):

    if noise_mechanism in ('dist_REP','dist_RUS','dist_RUS_reinit', 'Spin_X_RUS','Spin_Z_RUS','Spin_X_REP','Spin_Z_REP', 'Spin_depol_RUS','Spin_depol_REP', 
                           'single_emitter_dist_RUS', 'RUS_branching','RUS_blinking','RUS_blinking_reinit','RUS_branching_singledist'):
        
        blinked = blinking(num_fusions, num_physical_qbts, qbts_in_resource_states, qbts_in_fusions, A, D)
        
        primal_errors, dual_errors, lost_pr_pos, lost_du_pos = fusion_ErasureError(num_fusions, qbts_in_fusions, num_qubits_res_state, fusions_primal_isZZ, 
                                                                                   qubit_errors, fusions_layer_order, fusion_for_qubits, p_loss, p_err, 
                                                                                   dist_err, p_fusion_success_unbiased, p_err_unbiased, m, noise_mechanism,
                                                                                   blinked, loss_mask=loss_mask)

    
    elif noise_mechanism in ('RUS','REP','RUS_reinit'):
        primal_errors, dual_errors, lost_pr_pos, lost_du_pos = RUS_fusion_erasures(num_fusions, qbts_in_fusions, num_qubits_res_state, 
                                                                                   fusions_primal_isZZ, fusions_layer_order, fusion_for_qubits, p_loss, 
                                                                                   p_fusion_success_unbiased, m, noise_mechanism, loss_mask=loss_mask)

    elif noise_mechanism == 'RUS_blinking_reinit':
        blinked = blinking(num_fusions, num_physical_qbts, qbts_in_resource_states, qbts_in_fusions, A, D)
        
        primal_errors, dual_errors, lost_pr_pos, lost_du_pos = fusion_ErasureError(num_fusions, qbts_in_fusions, num_qubits_res_state, 
                                                                                   fusions_primal_isZZ, qubit_errors, fusions_layer_order, 
                                                                                   fusion_for_qubits, p_loss, p_err, dist_err, p_fusion_success_unbiased, 
                                                                                   p_err_unbiased, m, 'RUS_blinking_reinit', blinked, loss_mask=loss_mask)

    else:
        raise ValueError(f"Unsupported noise_mechanism: {noise_mechanism}")

    lost_pr, alive_pr, n_pr = compute_lost_and_alive_fusions(lost_pr_pos)
    lost_du, alive_du, n_du = compute_lost_and_alive_fusions(lost_du_pos)

    Hpr_dec, snd_pr_dec, new_log_pr = loss_correction(H_withlogop_primal, fus_syndr_mat_primal, lost_pr)
    Hdu_dec, snd_du_dec, new_log_du = loss_correction(H_withlogop_dual, fus_syndr_mat_dual, lost_du)

    if np.any(new_log_pr[lost_pr]) or np.any(new_log_du[lost_du]):
        return num_ec_runs_per_loss_trial

    if p_err_biased <= 1e-10 and p_err_unbiased <= 1e-10:
        return 0

    # Primal 
    if n_pr > 0:
        new_log_pr = new_log_pr[alive_pr]
        snd_pr_dec = snd_pr_dec[alive_pr]

        first_nonlost_pr = (Hpr_dec[:n_pr, lost_pr].any(axis=1)[::-1].argmax())
        first_nonlost_pr = n_pr - first_nonlost_pr

        NewH_unf_pr = Hpr_dec[first_nonlost_pr:-1, :][:, alive_pr]
        snd_pr_offset = snd_pr_dec - first_nonlost_pr

    else:
        NewH_unf_pr = Hpr_dec[:-1, :]
        new_log_pr = Hpr_dec[-1]
        snd_pr_offset = fus_syndr_mat_primal

    snd_pr_merged, new_ixs_pr, inv_ixs_pr, occ_pr, ok_pr = merge_multiedges_in_Hmat_faster(snd_pr_offset)
    NewH_pr = NewH_unf_pr[:, new_ixs_pr]
    new_log_pr = new_log_pr[new_ixs_pr]

    # Dual 
    if n_du > 0:
        new_log_du = new_log_du[alive_du]
        snd_du_dec = snd_du_dec[alive_du]

        first_nonlost_du = (Hdu_dec[:n_du, lost_du].any(axis=1)[::-1].argmax())
        first_nonlost_du = n_du - first_nonlost_du

        NewH_unf_du = Hdu_dec[first_nonlost_du:-1, :][:, alive_du]
        snd_du_offset = snd_du_dec - first_nonlost_du

    else:
        NewH_unf_du = Hdu_dec[:-1, :]
        new_log_du = Hdu_dec[-1]
        snd_du_offset = fus_syndr_mat_dual

    snd_du_merged, new_ixs_du, inv_ixs_du, occ_du, ok_du = merge_multiedges_in_Hmat_faster(snd_du_offset)
    NewH_du = NewH_unf_du[:, new_ixs_du]
    new_log_du = new_log_du[new_ixs_du]

    w_pr = np.ones(NewH_pr.shape[1]) if decoding_weights=='None' else None
    w_du = np.ones(NewH_du.shape[1]) if decoding_weights=='None' else None

    fails = 0
    for _ in range(num_ec_runs_per_loss_trial):
        # primal
        merged_pr = fusion_error_merged(new_ixs_pr, inv_ixs_pr, primal_errors)
        m_pr = Matching(NewH_pr, spacelike_weights=w_pr)
        fired_pr = (NewH_pr @ merged_pr) % 2
        corr_pr  = m_pr.decode(fired_pr)
        
        if (new_log_pr @ ((corr_pr + merged_pr) % 2)) & 1:
            fails += 1
            continue

        # dual
        merged_du = fusion_error_merged(new_ixs_du, inv_ixs_du, dual_errors)
        m_du = Matching(NewH_du, spacelike_weights=w_du)
        fired_du = (NewH_du @ merged_du) % 2
        corr_du  = m_du.decode(fired_du)
        
        if (new_log_du @ ((corr_du + merged_du) % 2)) & 1:
            fails += 1

    return fails

def decoder_successprob_error_vs_loss_single_vals_encoded( noise_vals, qubit_errors, loss_array, m, p_fusion_fail, qbts_in_fusions, 
                                                          num_qubits_res_state, fusions_primal_isZZ, fusions_layer_order, fusion_for_qubits, 
                                                          num_fusions, H_withlogop_primal, fus_syndr_mat_primal, H_withlogop_dual, fus_syndr_mat_dual,
                                                          num_loss_trials, num_ec_runs_per_loss_trial, noise_mechanism, decoding_weights, 
                                                          num_physical_qbts, qbts_in_resource_states, A, D):
    p_err, p_loss0, dist_err, br_err = noise_vals

    if noise_mechanism in ('RUS','RUS_reinit', 'Spin_X_RUS', 'Spin_Z_RUS', 'Spin_depol_RUS', 'Spin_depol_REP', 'single_emitter_dist_RUS', 
                           'RUS_blinking', 'RUS_blinking_reinit'):
        
        p_loss = p_loss0
        p_fusion_fail_unbiased    = p_fusion_fail
        p_fusion_fail_biased      = 0
        p_err_unbiased            = p_err
        p_err_biased              = p_err
        fusion_is_physical        = True
        p_fusion_success_unbiased = 1 - p_fusion_fail_unbiased
        p_fusion_success_biased   = 1 - p_fusion_fail_biased

    elif noise_mechanism in ('dist_RUS', 'dist_RUS_reinit'):
        p_loss = p_loss0
        p_fusion_success_unbiased = 0.5
        p_fusion_success_biased   = 1 - 0.5*p_err + 0.25*p_err**2
        p_err_unbiased            = p_err - 0.5*p_err**2
        p_err_biased              = 0
        fusion_is_physical        = True

    elif noise_mechanism == 'RUS_dist_SpinDepol':
        p_loss = p_loss0
        p_fusion_success_unbiased = 0.5
        p_fusion_success_biased   = 1 - 0.5*dist_err + 0.25*dist_err**2
        p_err_unbiased            = dist_err - 0.5*dist_err**2
        p_err_biased              = p_err
        fusion_is_physical        = True

    elif noise_mechanism in ('RUS_branching', 'RUS_branching_singledist'):
        p_loss = p_loss0
        p_fusion_fail_unbiased    = p_fusion_fail
        p_fusion_fail_biased      = 0
        p_err_unbiased            = br_err
        p_err_biased              = br_err
        fusion_is_physical        = True
        p_fusion_success_unbiased = 1 - p_fusion_fail_unbiased
        p_fusion_success_biased   = 1 - p_fusion_fail_biased

    else:
        raise ValueError(f"Unsupported noise_mechanism: {noise_mechanism}")

    orig_loss = (np.random.rand(num_loss_trials, qubit_errors.shape[1], m) < p_loss).astype(np.int8)

    num_errors = 0
    for trial in range(num_loss_trials):
        combined_loss = loss_array[trial] | orig_loss[trial]        
        num_errors += decoder_single_run_lossandEC_encoded(qubit_errors[trial], combined_loss, m, num_fusions, qbts_in_fusions, num_qubits_res_state,
                                                           fusions_primal_isZZ, fusions_layer_order, fusion_for_qubits, H_withlogop_primal, 
                                                           fus_syndr_mat_primal, H_withlogop_dual, fus_syndr_mat_dual, p_loss, p_fusion_success_biased, 
                                                           p_fusion_success_unbiased, p_err, p_err_biased, p_err_unbiased, dist_err, num_physical_qbts, 
                                                           qbts_in_resource_states, A, D, num_ec_runs_per_loss_trial=num_ec_runs_per_loss_trial, 
                                                           noise_mechanism=noise_mechanism, decoding_weights=decoding_weights, fusion_is_physical=fusion_is_physical)

    return num_errors / (num_loss_trials*num_ec_runs_per_loss_trial)

def decoder_successprob_error_vs_loss_list_parallelized_encoded(error_vs_loss_list, p_fusion_fail, L, m, A, D, log_op_axis='z', num_loss_trials=1000, 
                                                                num_ec_runs_per_loss_trial=1, noise_mechanism='RUS', noise_fits=None, 
                                                                decoding_weights='None'): 

    if log_op_axis == 'x':
        log_op_ix = 0
    elif log_op_axis == 'y':
        log_op_ix = 1
    elif log_op_axis == 'z':
        log_op_ix = 2
    else:
        raise ValueError('log_op_axis needs to be in [x, y, z]')
    Lattice = FFCCLattice_chain(L, L, L)    

    num_fusions = Lattice.num_fusions
    qbts_in_fusions = Lattice.qbts_in_fusions
    qbts_in_resource_states = Lattice.qbts_in_resource_states
    num_res_states = Lattice.num_res_states
    num_physical_qbts = Lattice.num_physical_qubits
    Lattice.get_fusions_primal_isZZ()
    fusions_primal_isZZ = Lattice.fusions_primal_isZZ
    num_qubits_res_state = Lattice.num_qubits_res_state*Lattice.lattice_z_size 
    
    if noise_mechanism == 'RUS' or noise_mechanism == 'dist_RUS_reinit' or noise_mechanism == 'RUS_blinking_reinit' or noise_mechanism == 'RUS_dist_SpinDepol': 
        Lattice.get_fusions_layer_order() # fusion indices layer by layer
        fusions_layer_order = Lattice.fusions_layer_order
        fusion_for_qubits = Lattice.fusion_for_qubits
    else:
        fusions_layer_order = None
        fusion_for_qubits = None

    log_op_primal_fuss0 = Lattice.log_ops_fusions[log_op_ix]
    log_op_array_primal = np.array([1 if x in log_op_primal_fuss0 else 0 for x in range(Lattice.num_fusions)], dtype=np.uint8)
    H_matrix_primal = Lattice.get_matching_matrix(lattice_type='Primal')
    fus_syndr_mat_primal = np.where(H_matrix_primal.T)[1].reshape((Lattice.num_fusions, 2)).astype(dtype=np.int32)
    H_withlogop_primal = np.vstack([H_matrix_primal, log_op_array_primal])

    log_op_dual_fuss0 = Lattice.log_ops_fusions_dual[log_op_ix]
    log_op_array_dual = np.array([1 if x in log_op_dual_fuss0 else 0 for x in range(Lattice.num_fusions)], dtype=np.uint8)
    H_matrix_dual = Lattice.get_matching_matrix(lattice_type='Dual')
    fus_syndr_mat_dual = np.where(H_matrix_dual.T)[1].reshape((Lattice.num_fusions, 2)).astype(dtype=np.int32)
    H_withlogop_dual = np.vstack([H_matrix_dual, log_op_array_dual])
    
    print('Simulating errors...')
    n_samples = num_loss_trials 
    n_steps = error_vs_loss_list.shape[0] 
    loss_array_list = np.zeros((n_steps, n_samples, num_qubits_res_state*num_res_states, m), dtype=np.int8)

    if noise_mechanism in ('Spin_depol_RUS'):     
        qubit_errors_list = np.zeros((n_steps, n_samples, num_qubits_res_state*num_res_states, m, 2), dtype=np.int8)
        loss_array_list = np.zeros((n_steps, n_samples, num_qubits_res_state*num_res_states, m), dtype=np.int8)
    
        for i in range(n_steps):
            p = error_vs_loss_list[i][0]
            qe, la = sampling_qubit_errors_spindepol(p, num_res_states, m, num_qubits_res_state, qbts_in_resource_states, n_samples)
            qubit_errors_list[i] = qe
            loss_array_list[i]  = la

    elif noise_mechanism == 'RUS_dist_SpinDepol':
        if not np.any(error_vs_loss_list[:,0] > 1.e-10):
            qubit_errors_list = np.zeros(shape=(np.shape(error_vs_loss_list)[0], n_samples, num_qubits_res_state*num_res_states, m, 2), dtype=np.int8) 
        
        else:
            qubit_errors_list = np.zeros(shape=(np.shape(error_vs_loss_list)[0], n_samples, num_qubits_res_state*num_res_states, m, 2), dtype=np.int8)
            for i in range(np.shape(error_vs_loss_list)[0]):
                qubit_errors_list[i,:,:,:,:] = sampling_qubit_errors(error_vs_loss_list[i][0]/3, error_vs_loss_list[i][0]/3, error_vs_loss_list[i][0]/3,
                                                                     num_res_states, m, num_qubits_res_state, qbts_in_resource_states, n_samples)

    elif noise_mechanism == 'RUS_branching':
        qubit_errors_list = np.zeros(shape=(np.shape(error_vs_loss_list)[0], n_samples, num_qubits_res_state*num_res_states, m, 2), dtype=np.int8)
        for i in range(np.shape(error_vs_loss_list)[0]):
            qubit_errors_list[i,:,:,:,:] = sampling_qubit_errors_branching(error_vs_loss_list[i][3], num_res_states, m, num_qubits_res_state,
                                                                           qbts_in_resource_states, n_samples)
    
    elif noise_mechanism == 'single_emitter_dist_RUS': 
        qubit_errors_list = np.zeros(shape=(np.shape(error_vs_loss_list)[0], n_samples, num_qubits_res_state*num_res_states, m, 2), dtype=np.int8)
        for i in range(np.shape(error_vs_loss_list)[0]):
            qubit_errors_list[i,:,:,:,:] = sampling_qubit_errors_single_emitter_dist(error_vs_loss_list[i][0], num_res_states, m, num_qubits_res_state,
                                                                 qbts_in_resource_states, n_samples)
    
    elif noise_mechanism == 'RUS_blinking' or noise_mechanism == 'RUS_blinking_reinit':
        qubit_errors_list = np.zeros(shape=(np.shape(error_vs_loss_list)[0], n_samples,num_qubits_res_state*num_res_states, m, 2), dtype=np.int8)

    
    elif noise_mechanism == 'RUS_branching_singledist':
        qubit_errors_list = np.zeros(shape=(np.shape(error_vs_loss_list)[0], n_samples, num_qubits_res_state*num_res_states, m, 2), dtype=np.int8)
        for i in range(np.shape(error_vs_loss_list)[0]):
            qubit_errors_list[i,:,:,:,:] = sampling_qubit_errors_branching_singledist(error_vs_loss_list[i][0], error_vs_loss_list[i][3], num_res_states, m,
                                                                                      num_qubits_res_state, qbts_in_resource_states, num_loss_trials)
    
    else:
        qubit_errors_list = np.empty(shape=(np.shape(error_vs_loss_list)[0],n_samples,2), dtype=np.int8)

    print('Start pooling...')
    pool = multiprocessing.Pool()
    success_probs = pool.starmap(
        decoder_successprob_error_vs_loss_single_vals_encoded, zip(error_vs_loss_list, qubit_errors_list, loss_array_list, repeat(m), 
                                                                   repeat(p_fusion_fail), repeat(qbts_in_fusions), repeat(num_qubits_res_state), 
                                                                   repeat(fusions_primal_isZZ), repeat(fusions_layer_order), repeat(fusion_for_qubits),
                                                                   repeat(num_fusions), repeat(H_withlogop_primal), repeat(fus_syndr_mat_primal),
                                                                   repeat(H_withlogop_dual), repeat(fus_syndr_mat_dual), repeat(num_loss_trials),
                                                                   repeat(num_ec_runs_per_loss_trial), repeat(noise_mechanism), repeat(decoding_weights),
                                                                   repeat(num_physical_qbts), repeat(qbts_in_resource_states), repeat(A), repeat(D)))

    return np.array(success_probs)



#################### Functions that support the decoder ####################

# Combine lost and alive fusion computations for both primal and dual
def compute_lost_and_alive_fusions(lost_fusions_pos):
    lost_fusions = np.flatnonzero(lost_fusions_pos).astype(np.int32)
    alive_fusions = np.flatnonzero(~np.array(lost_fusions_pos, dtype=bool)).astype(np.int32)
    return lost_fusions, alive_fusions, len(lost_fusions)

# Helper function for loss correction
def loss_correction(H_withlogop, fus_syndr_mat, lost_fusions):
    if lost_fusions.size > 0:
        # Decoding
        H_withlogop_dec, fus_syndr_mat_dec = LossDecoder_GaussElimin_trackqbts_noorderedlost(H_withlogop, fus_syndr_mat, lost_fusions)
        new_logop = H_withlogop_dec[-1]
        
    else:
        H_withlogop_dec = H_withlogop
        fus_syndr_mat_dec = fus_syndr_mat
        new_logop = H_withlogop[-1]
        
    return H_withlogop_dec, fus_syndr_mat_dec, new_logop

def process_corrections(num_lost_fusions, alive_fusions, lost_fusions, H_withlogop_dec, fus_syndr_mat_dec, fus_syndr_mat, new_logop):
    if num_lost_fusions > 0:
        alive = np.array(alive_fusions, dtype=np.int32)
        lost = np.array(lost_fusions, dtype=np.int32)

        new_logop = new_logop[alive]
        new_fus_syndr_mat_dec = fus_syndr_mat_dec[alive]

        # Find the first non-lost syndrome index
        first_nonlostsyndr = (np.any(H_withlogop_dec[:num_lost_fusions, lost], axis=1)[::-1]).argmax()
        first_nonlostsyndr = num_lost_fusions - first_nonlostsyndr

        NewH_unfiltered = H_withlogop_dec[first_nonlostsyndr:-1, alive]
        new_fus_syndr_mat_dec_ = new_fus_syndr_mat_dec - first_nonlostsyndr
        
    else:
        NewH_unfiltered = H_withlogop_dec[:-1]
        new_logop = H_withlogop_dec[-1]
        new_fus_syndr_mat_dec_ = fus_syndr_mat

    # Merge multi-edges
    (
        new_fus_syndr_mat_dec,
        new_ixs,
        inverse_ixs,
        occ_counts,
        has_no_zeroed_fus,) = merge_multiedges_in_Hmat_faster(new_fus_syndr_mat_dec_)

    new_fus_syndr_mat_dec = np.array(new_fus_syndr_mat_dec, dtype=np.uint32)
    NewH = NewH_unfiltered[:, new_ixs]
    new_logop = new_logop[new_ixs]

    return NewH, new_logop, new_fus_syndr_mat_dec, new_ixs, inverse_ixs

def propagate_errors_multiedge_PY_vectorized(num_syndromes, num_multiedges, syndromes_of_multiedge, multiedge_of_fusions, errors_in_fusions):
    fired_syndromes = np.zeros(num_syndromes, dtype=np.uint8)
    errors_in_multiedges = np.zeros(num_multiedges, dtype=np.uint8)

    # Filter valid multiedges and their associated errors
    valid_mask = (0 <= multiedge_of_fusions) & (multiedge_of_fusions < num_multiedges) & (errors_in_fusions[:multiedge_of_fusions.shape[0]] != 0)
    valid_multiedges = multiedge_of_fusions[valid_mask]

    # Gather associated syndromes for valid multiedges
    syndromes_1 = syndromes_of_multiedge[valid_multiedges, 0]
    syndromes_2 = syndromes_of_multiedge[valid_multiedges, 1]

    # Toggle the states of the syndromes
    np.bitwise_xor.at(fired_syndromes, syndromes_1, 1)  # XOR toggling
    np.bitwise_xor.at(fired_syndromes, syndromes_2, 1)

    # Toggle the error states of the multiedges
    np.bitwise_xor.at(errors_in_multiedges, valid_multiedges, 1)

    return fired_syndromes, errors_in_multiedges

# Function for swapping every 2 rows: [0,1,2,3] -> [1,0,3,2]
def Swap2Rows(array): 
    for i in range(int(len(array)/2)):
        array[[2*i, 2*i+1]] = array[[2*i+1, 2*i]]
        
    return array 

"""
original smapling function from M.-L. Chan et al. without time-bin (where spin error in between pulses can happen and e.g. cause loss)
error_px,error_py,error_pz: error rates on the spin
num_res_states: number of resource states (i.e. spins in the lattice)
m: number of qubits per logical qubit (number of branches +1 of branched chain)
num_qubits_res_state: number of logical qubits per resource state (i.e. length of branched chain)
qbts_in_resource_states: (num_res_states x num_qubits_res_state) array with indices of logical qubits
num_loss_trials: number of repetitions for averaging
"""
def sampling_qubit_errors(error_px,error_py,error_pz,num_res_states,m,num_qubits_res_state,qbts_in_resource_states,num_loss_trials):
    n_samples = num_loss_trials
    num_logqts_res_state = num_qubits_res_state

    sampled_errors = encoded_chain_sampler(num_logqts_res_state, error_px, error_py, error_pz, n_samples, num_res_states, code_size=m, 
                                           extra_chain_length=6, xx_eras_rate=0.0, zz_eras_rate=0.0)

    sampled_photons_error_list = np.zeros(shape=(n_samples, num_res_states, num_logqts_res_state,m,2)).astype(dtype=np.int8)

    qubit_errors_ = np.zeros(shape=(n_samples,num_res_states*num_logqts_res_state,m,2)).astype(dtype=np.int8)
    
    # Combine X and Z errors into a single list for a qubit
    for res_state_ix in range(num_res_states):
        reshaped_errors_0 = np.transpose(sampled_errors[0][res_state_ix]).reshape(n_samples, num_logqts_res_state, m, 1)
        reshaped_errors_1 = np.transpose(sampled_errors[1][res_state_ix]).reshape(n_samples, num_logqts_res_state, m, 1)
        sampled_photons_error_list[:, res_state_ix] = np.concatenate((reshaped_errors_0, reshaped_errors_1), axis=3)
        
    # Create a mapping from qbts_in_resource_states to their corresponding indices
    for res_state_ix in range(num_res_states):
        logqbt_indices = qbts_in_resource_states[res_state_ix]
        qubit_errors_[:, logqbt_indices] = sampled_photons_error_list[:, res_state_ix]

    swap_indices = Swap2Rows(np.arange(num_logqts_res_state * num_res_states))
    qubit_errors = qubit_errors_[:, swap_indices] 
    
    return qubit_errors

"""
error_pz: dephasing error probability on the spin
num_res_states: number of resource states (i.e. spins in the lattice)
m: number of qubits per logical qubit (number of branches +1 of branched chain)
num_qubits_res_state: number of logical qubits per resource state (i.e. length of branched chain)
qbts_in_resource_states: (num_res_states x num_qubits_res_state) array with indices of logical qubits
n_sample: number of repetitions for averaging
"""
def sampling_qubit_errors_single_emitter_dist(error_pz, num_res_states, m, num_qubits_res_state, qbts_in_resource_states, n_sample): 
    N = (num_qubits_res_state * m) + 1
    n_CNOT = N - 1
    error_pz_all = np.random.binomial(1, error_pz, size=(num_res_states, n_sample, n_CNOT)).astype(np.uint8)
    
    sampled_errors = optimized_simulate_many_resources_dist(num_qubits_res_state, error_pz_all, m)
    num_logqts_res_state = num_qubits_res_state
    n_rs = num_res_states
    X_err = np.transpose(sampled_errors[0], (2, 0, 1))
    Z_err = np.transpose(sampled_errors[1], (2, 0, 1))
    X_err = X_err.reshape(n_sample, n_rs, num_logqts_res_state, m, 1)
    Z_err = Z_err.reshape(n_sample, n_rs, num_logqts_res_state, m, 1)
    
    sampled_photons_error_list = np.concatenate((X_err, Z_err), axis=4)
    
    qubit_errors_ = np.zeros((n_sample, n_rs*num_logqts_res_state, m, 2), dtype=np.uint8)
    
    for res_state_ix in range(num_res_states):
        logqbt_indices = qbts_in_resource_states[res_state_ix]
        qubit_errors_[:, logqbt_indices] = sampled_photons_error_list[:, res_state_ix]
    
    swap_indices = Swap2Rows(np.arange(num_res_states*num_logqts_res_state))
    qubit_errors = qubit_errors_[:, swap_indices]
    
    return qubit_errors   


"""
dominant_branching_error: error rate for the case of branching where one ends up with X on spin and X on photon (see Fig. 8a in https://arxiv.org/pdf/2507.16152)
num_res_states: number of resource states (i.e. spins in the lattice)
m: number of qubits per logical qubit (number of branches +1 of branched chain)
num_qubits_res_state: number of logical qubits per resource state (i.e. length of branched chain)
qbts_in_resource_states: (num_res_states x num_qubits_res_state) array with indices of logical qubits
n_sample: number of repetitions for averaging
"""
def sampling_qubit_errors_branching(dominant_branching_error, num_res_states, m, num_qubits_res_state, qbts_in_resource_states, n_sample): 
    N = (num_qubits_res_state * m) + 1
    n_CNOT = N - 1
    branching_all = np.random.binomial(1, dominant_branching_error, size=(num_res_states, n_sample, n_CNOT)).astype(np.uint8)
    
    sampled_errors = optimized_simulate_many_resources(num_qubits_res_state, branching_all, m)
    num_logqts_res_state = num_qubits_res_state
    n_rs = num_res_states
    X_err = np.transpose(sampled_errors[0], (2, 0, 1))
    Z_err = np.transpose(sampled_errors[1], (2, 0, 1))
    X_err = X_err.reshape(n_sample, n_rs, num_logqts_res_state, m, 1)
    Z_err = Z_err.reshape(n_sample, n_rs, num_logqts_res_state, m, 1)
    
    sampled_photons_error_list = np.concatenate((X_err, Z_err), axis=4)
    
    qubit_errors_ = np.zeros((n_sample, n_rs*num_logqts_res_state, m, 2), dtype=np.uint8)
    
    for res_state_ix in range(num_res_states):
        logqbt_indices = qbts_in_resource_states[res_state_ix]
        qubit_errors_[:, logqbt_indices] = sampled_photons_error_list[:, res_state_ix]
    
    swap_indices = Swap2Rows(np.arange(num_res_states*num_logqts_res_state))
    qubit_errors = qubit_errors_[:, swap_indices]
    
    return qubit_errors


"""
error_pz: Z error on photon due to distinguishability (single emitter)
dominant_branching_error: error rate for the case of branching where one ends up with X on spin and X on photon (see Fig. 8a in https://arxiv.org/pdf/2507.16152)
num_res_states: number of resource states (i.e. spins in the lattice)
m: number of qubits per logical qubit (number of branches +1 of branched chain)
num_qubits_res_state: number of logical qubits per resource state (i.e. length of branched chain)
qbts_in_resource_states: (num_res_states x num_qubits_res_state) array with indices of logical qubits
n_sample: number of repetitions for averaging
"""
def sampling_qubit_errors_branching_singledist(error_pz, dominant_branching_error, num_res_states, m, num_qubits_res_state, qbts_in_resource_states, 
                                               n_sample): 

    N = (num_qubits_res_state * m) + 1
    n_CNOT = N - 1
    branching_all = np.random.binomial(1, dominant_branching_error, size=(num_res_states, n_sample, n_CNOT)).astype(np.uint8)
    error_pz_all = np.random.binomial(1, error_pz, size=(num_res_states, n_sample, n_CNOT)).astype(np.uint8)
    
    sampled_errors = optimized_simulate_many_resources_branching_singledist(num_qubits_res_state, error_pz_all, branching_all, m)
    num_logqts_res_state = num_qubits_res_state
    n_rs = num_res_states
    X_err = np.transpose(sampled_errors[0], (2, 0, 1))
    Z_err = np.transpose(sampled_errors[1], (2, 0, 1))
    X_err = X_err.reshape(n_sample, n_rs, num_logqts_res_state, m, 1)
    Z_err = Z_err.reshape(n_sample, n_rs, num_logqts_res_state, m, 1)
    
    sampled_photons_error_list = np.concatenate((X_err, Z_err), axis=4)
    
    qubit_errors_ = np.zeros((n_sample, n_rs*num_logqts_res_state, m, 2), dtype=np.uint8)
    
    for res_state_ix in range(num_res_states):
        logqbt_indices = qbts_in_resource_states[res_state_ix]
        qubit_errors_[:, logqbt_indices] = sampled_photons_error_list[:, res_state_ix]
    
    swap_indices = Swap2Rows(np.arange(num_res_states*num_logqts_res_state))
    qubit_errors = qubit_errors_[:, swap_indices]
    
    return qubit_errors

"""
Applies blinking errors to each qubit in order throughout the lattice, then checks which fusions have been lost as a result
num_fusions: number of all fusions
fusion_erasures: all fusions that have erasures
num_physical_qbts: number of physical qubits in the lattice
qbts_in_resource_states: number of qubits in the resource state
qbts_in_fusions: mapping to get which fusion the qubit belongs to
A, D: parameters for blinking. A: turn on rate, D: turn off rate
"""
def blinking(num_fusions, num_physical_qbts, qbts_in_resource_states, qbts_in_fusions, A, D):
    fusion_erasures = np.zeros((num_fusions, 2), dtype=np.uint8)
    
    # Allocate space for final 0/1 (alive/dead) states
    is_dead_all = np.zeros(num_physical_qbts, dtype=np.uint8)

    # Generate "alive-to-dead" & "dead-to-alive" random events across all qubits.
    alive_to_dead_all = np.random.binomial(1, D, size=num_physical_qbts)  # 1 => alive->dead transition
    dead_to_alive_all = np.random.binomial(1, A, size=num_physical_qbts)  # 1 => dead->alive transition

    # Stationary distribution probabilities (for the first qubit in the extended chain).
    if A + D > 0:
        pi_alive = A / (A + D)
        pi_dead = D / (A + D)
    else:
        # Degenerate case: if A = D = 0, pick 50/50
        pi_alive = 0.5
        pi_dead = 0.5

    # Initialize the *first* qubit in the extended chain via the stationary distribution
    is_dead_all[0] = np.random.choice([0, 1], p=[pi_alive, pi_dead])

    # Precompute the positions at which transitions occur
    idx_a2d = np.flatnonzero(alive_to_dead_all)  # Potential 0->1 transitions
    idx_d2a = np.flatnonzero(dead_to_alive_all)  # Potential 1->0 transitions

    # Initialize pointers for transition indices
    ptr_a2d = 0
    ptr_d2a = 0

    # Initialize current state and position
    current_state = is_dead_all[0]
    current_pos = 0

    # Main loop: fill the chain from left (0) to right (total_qbts)
    while current_pos < num_physical_qbts:
        if current_state == 0:
            # Find the next alive-to-dead transition
            while ptr_a2d < len(idx_a2d) and idx_a2d[ptr_a2d] < current_pos:
                ptr_a2d += 1
            if ptr_a2d >= len(idx_a2d):
                is_dead_all[current_pos:] = 0
                break
            next_dead = idx_a2d[ptr_a2d]
            if next_dead >= num_physical_qbts:
                is_dead_all[current_pos:] = 0
                break
            is_dead_all[current_pos:next_dead] = 0
            is_dead_all[next_dead] = 1
            current_state = 1
            current_pos = next_dead + 1
            ptr_a2d += 1
        else:
            # Find the next dead-to-alive transition
            while ptr_d2a < len(idx_d2a) and idx_d2a[ptr_d2a] < current_pos:
                ptr_d2a += 1
            if ptr_d2a >= len(idx_d2a):
                is_dead_all[current_pos:] = 1
                break
            next_alive = idx_d2a[ptr_d2a]
            if next_alive >= num_physical_qbts:
                is_dead_all[current_pos:] = 1
                break
            is_dead_all[current_pos:next_alive] = 1
            is_dead_all[next_alive] = 0
            current_state = 0
            current_pos = next_alive + 1
            ptr_d2a += 1

    lost_qbts = np.zeros(num_physical_qbts, dtype=np.uint8)
    lost_qbts[is_dead_all == 1] = 1
    
    for fus_ix, (qbt1, qbt2) in enumerate(qbts_in_fusions):
        # Checks if either qubit is lost
        if lost_qbts[qbt1] or lost_qbts[qbt2]:
            # Marks fusion as lost if either qubit is lost
            fusion_erasures[fus_ix] = [1, 1]

    return fusion_erasures

#################### T2 time-bin simulation ####################

@numba.njit
def propagate_errors(pauli_labels, late):
    pauli_dx = np.array([0, 1, 0, 1], dtype=np.int8)
    pauli_dz = np.array([0, 0, 1, 1], dtype=np.int8)
    ep = np.zeros((2, 3), np.int8)
    x_between = False

    for gate in range(1, 5):
        if gate == 1 and not late:
            ep = controlledNOT(ep, 0, 1)
        if gate == 3 and late:
            ep = controlledNOT(ep, 0, 1)

        label = pauli_labels[gate - 1]
        dx = pauli_dx[label]
        dz = pauli_dz[label]
        ep[0, 0] ^= dx
        ep[0, 1] ^= dz

        # Late-bin X at gates 1–2 ⇒ loss
        if gate <= 2 and dx == 1 and late:
            ep[1, 2] = 1

        # Early-bin X at gates 2–3 ⇒ extra Z later
        if not late and gate in (2, 3) and dx == 1:
            x_between = True

    if not late and x_between:
        ep[0, 1] ^= 1

    return ep

@numba.njit
def simulate_one_resource_state_jit_spindepol(ep, loss_array, pauli_labels_all, late_all, hadamard_labels, n_CNOT, m):
    pauli_dx = np.array([0, 1, 0, 1], dtype=np.int8)
    pauli_dz = np.array([0, 0, 1, 1], dtype=np.int8)
    h_idx = 0
    hadamard(ep, 0)
    label = hadamard_labels[h_idx]
    dx = pauli_dx[label]
    dz = pauli_dz[label]
    ep[0, 0] ^= dx
    ep[0, 1] ^= dz
    h_idx += 1

    for i in range(n_CNOT):
        if not late_all[i]:
            ep = controlledNOT(ep, 0, i+1)
        else:
            ep = controlledNOT(ep, 0, i+1)

        block_ep = propagate_errors(pauli_labels_all[i], late_all[i])

        # record loss
        loss_array[i] = block_ep[1, 2]
        # accumulate spin
        ep[0, 0] ^= block_ep[0, 0]
        ep[0, 1] ^= block_ep[0, 1]
        # photon flags
        ep[i+1, 0] = block_ep[1, 0]
        ep[i+1, 1] = block_ep[1, 1]

        # periodic H + injection from hadamard_labels
        if (i + 1) % m == 0 and i < n_CNOT - 1:
            hadamard(ep, 0)
            label = hadamard_labels[h_idx]
            dx = pauli_dx[label]
            dz = pauli_dz[label]
            ep[0, 0] ^= dx
            ep[0, 1] ^= dz
            h_idx += 1

    return ep

@numba.njit(parallel=True)
def run_many_resource_states_jit_spindepol(Xbig, Zbig, loss_big, pauli_labels_big, late_big, hadamard_labels_big, n_CNOT, m):
    """
    Runs n_rs × n_sample resource‐state sims in parallel.
    """
    n_rs, N, n_sample = Xbig.shape
    
    for i in numba.prange(n_rs):
        for s in range(n_sample):
            ep         = np.zeros((N, 2), dtype=np.uint8)
            loss_array = loss_big[i, s]      
            pauli_labels_all = pauli_labels_big[i, s]   
            late_all         = late_big[i, s]   
            had_labels       = hadamard_labels_big[i, s]            
            simulate_one_resource_state_jit_spindepol(ep, loss_array, pauli_labels_all, late_all, had_labels, n_CNOT, m)
            
            for q in range(N):
                Xbig[i, q, s] = ep[q, 0]
                Zbig[i, q, s] = ep[q, 1]
    
    return Xbig, Zbig, loss_big

def optimized_simulate_many_resources_spindepol(num_qubits_res_state, pauli_labels_big, late_big, hadamard_labels_big, m):
    """
    Vectorized simulation over all resource states.
    """
    n_rs, n_sample, n_CNOT, _ = pauli_labels_big.shape
    N = num_qubits_res_state * m + 1

    Xbig = np.zeros((n_rs, N, n_sample), dtype=np.uint8)
    Zbig = np.zeros_like(Xbig)
    loss_big = np.zeros((n_rs, n_sample, n_CNOT), dtype=np.uint8)
    Xbig, Zbig, loss_big = run_many_resource_states_jit_spindepol(Xbig, Zbig, loss_big, pauli_labels_big, late_big, hadamard_labels_big, n_CNOT, m)

    # Drop spin qubit
    X_error = Xbig[:, 1:, :]
    Z_error = Zbig[:, 1:, :]

    return X_error, Z_error, loss_big

def sampling_qubit_errors_spindepol(p_error, num_res_states, m, num_qubits_res_state, qbts_in_resource_states, n_sample):
    N      = num_qubits_res_state * m + 1
    n_CNOT = N - 1
    n_rs   = num_res_states

    # number of Hadamards: initial one + one after every m-th CNOT
    n_H = 1 + (n_CNOT - 1)//m

    # Pre-sample all Pauli labels {0:I,1:X,2:Z,3:Y}
    probs = [1 - p_error, p_error/3, p_error/3, p_error/3]
    pauli_labels_big = np.random.choice([0,1,2,3], size=(n_rs, n_sample, n_CNOT, 4), p=probs).astype(np.uint8)
    hadamard_labels_big = np.random.choice([0,1,2,3], size=(n_rs, n_sample, n_H), p=probs).astype(np.uint8)
    late_big = (np.random.rand(n_rs, n_sample, n_CNOT) < 0.5)
    X_error_big, Z_error_big, loss_big = optimized_simulate_many_resources_spindepol(num_qubits_res_state, pauli_labels_big, late_big, 
                                                                                     hadamard_labels_big, m)

    X_err = np.transpose(X_error_big, (2, 0, 1))   # → (n_sample, n_rs, N-1)
    Z_err = np.transpose(Z_error_big, (2, 0, 1))   # → (n_sample, n_rs, N-1)
    X_err = X_err.reshape(n_sample, n_rs, num_qubits_res_state, m, 1)
    Z_err = Z_err.reshape(n_sample, n_rs, num_qubits_res_state, m, 1)
    sampled_photons_error_list = np.concatenate((X_err, Z_err), axis=4)

    # assemble qubit_errors_
    qubit_errors_ = np.zeros((n_sample, n_rs*num_qubits_res_state, m, 2), dtype=np.uint8)
    for rs in range(num_res_states):
        locs = qbts_in_resource_states[rs]
        qubit_errors_[:, locs] = sampled_photons_error_list[:, rs]

    swap_indices = Swap2Rows(np.arange(num_res_states*num_qubits_res_state))
    qubit_errors = qubit_errors_[:, swap_indices]

    loss_arr = loss_big.transpose(1, 0, 2)  
    loss_array = loss_arr.reshape(n_sample, num_res_states*num_qubits_res_state, m)

    return qubit_errors, loss_array

#################### Other simulations ####################

@numba.njit
def hadamard(ep, q):
    """
    Swap X <-> Z only if exactly one is set (i.e., if X ^ Z == 1)
    """
    x = ep[q, 0]
    z = ep[q, 1]
    if x ^ z == 1:
        ep[q, 0] = z
        ep[q, 1] = x

    return ep

@numba.njit
def controlledNOT(ep, c, t):
    """
    X-error on control => X error is copied to target (control error remains).
    Z-error on control => Z error is removed from control and copied to target.
    """
    # If there is an X error on the control (column 0), copy it to the target
    if ep[c, 0] == 1:
        ep[t, 0] ^= 1

    # If there is a Z error on the control (column 1), transfer it:
    if ep[c, 1] == 1:
        ep[t, 1] ^= 1  # Copy the Z error to the target
        ep[c, 1] = 0   # Remove the Z error from the control

    return ep
    
@numba.njit
def simulate_one_resource_state_jit(ep, branching, n_CNOT, m):
    """
    Run one resource state simulation for branching.
    """
    # Hadamard on spin qubit 0
    hadamard(ep, 0)
    
    for i in range(n_CNOT):
        controlledNOT(ep, 0, i+1)

        # branching error
        if branching[i] == 1:
            if i == 0: # first photon gets no spin X error
                ep[i+1, 0] ^= 1  # flip X on photon
            else:
                ep[0, 0] ^= 1  # flip X on spin
                ep[i+1, 0] ^= 1  # flip X on photon too

        # Hadamard after every m-th
        if ((i + 1) % m == 0) and (i < n_CNOT - 1):
            hadamard(ep, 0)

@numba.njit(parallel=True)
def run_many_resource_states_jit(Xbig, Zbig, n_CNOT, m, branching_all):
    """
    Simulate all resource states and samples in one go for branching.
    """
    n_rs = Xbig.shape[0]
    N = Xbig.shape[1]
    n_sample = Xbig.shape[2]
    
    for i in numba.prange(n_rs):
        for s in range(n_sample):
            ep = np.zeros((N, 2), dtype=np.uint8)
            simulate_one_resource_state_jit(ep, branching_all[i, s], n_CNOT, m)
            for q in range(N):
                Xbig[i, q, s] = ep[q, 0]
                Zbig[i, q, s] = ep[q, 1]

    return Xbig, Zbig

def optimized_simulate_many_resources(num_qubits_res_state, branching_all, m):
    """
    Vectorized simulation over all resource states for branching.
    """
    n_rs = branching_all.shape[0]
    n_sample = branching_all.shape[1]
    N = (num_qubits_res_state * m) + 1 # Total number of qubits per resource state
    n_CNOT = N - 1
    
    Xbig = np.zeros((n_rs, N, n_sample), dtype=np.uint8)
    Zbig = np.zeros((n_rs, N, n_sample), dtype=np.uint8)
    
    run_many_resource_states_jit(Xbig, Zbig, n_CNOT, m, branching_all)
    
    # Remove spin qubit
    X_error = Xbig[:, 1:, :]
    Z_error = Zbig[:, 1:, :]
    
    return [X_error, Z_error]
        
@numba.njit
def simulate_one_resource_state_jit_dist(ep, z_error, n_CNOT, m):
    """
    Run one resource state simulation for single-emitter distinguishability.
    """
    # Hadamard on spin qubit 0
    hadamard(ep, 0)
    
    for i in range(n_CNOT):
        controlledNOT(ep, 0, i+1)

        if z_error[i] == 1: # exclude first photon
            ep[i+1, 1] ^= 1  # flip Z on the ith photon

        # Hadamard after every m-th
        if ((i + 1) % m == 0) and (i < n_CNOT - 1):
            hadamard(ep, 0)

@numba.njit(parallel=True)
def run_many_resource_states_jit_dist(Xbig, Zbig, n_CNOT, m, z_error_all):
    """
    Simulate all resource states and samples in one go for single-emitter distinguishability.
    """
    n_rs = Xbig.shape[0]
    N = Xbig.shape[1]
    n_sample = Xbig.shape[2]
    
    for i in numba.prange(n_rs):
        for s in range(n_sample):
            ep = np.zeros((N, 2), dtype=np.uint8)
            simulate_one_resource_state_jit_dist(ep, z_error_all[i, s], n_CNOT, m)
            for q in range(N):
                Xbig[i, q, s] = ep[q, 0]
                Zbig[i, q, s] = ep[q, 1]

    return Xbig, Zbig

def optimized_simulate_many_resources_dist(num_qubits_res_state, z_error_all, m):
    """
    Vectorized simulation over all resource states for single-emitter distinguishability.
    """
    n_rs = z_error_all.shape[0]
    n_sample = z_error_all.shape[1]
    N = (num_qubits_res_state * m) + 1  
    n_CNOT = N - 1
    
    Xbig = np.zeros((n_rs, N, n_sample), dtype=np.uint8)
    Zbig = np.zeros((n_rs, N, n_sample), dtype=np.uint8)
    
    run_many_resource_states_jit_dist(Xbig, Zbig, n_CNOT, m, z_error_all)
    
    # Remove spin qubit
    X_error = Xbig[:, 1:, :]
    Z_error = Zbig[:, 1:, :]
    
    return [X_error, Z_error]

@numba.njit
def simulate_one_resource_state_jit_br_singledist(ep, error_pz, branching, n_CNOT, m):
    """
    Run one resource state simulation for branching AND single-emitter distinguishability.
    """
    # Hadamard on spin qubit 0
    hadamard(ep, 0)
    
    for i in range(n_CNOT):
        controlledNOT(ep, 0, i+1)

        # branching error
        if branching[i] == 1:
            if i == 0: # first photon gets no spin X error
                ep[i+1, 0] ^= 1  # flip X on photon
            else:
                ep[0, 0] ^= 1  # flip X on spin
                ep[i+1, 0] ^= 1  # flip X on photon too

        # single-emitter dist error
        if error_pz[i] == 1:
            ep[i+1, 1] ^= 1  # flip Z on the ith photon

        # Hadamard after every m-th
        if ((i + 1) % m == 0) and (i < n_CNOT - 1):
            hadamard(ep, 0)

@numba.njit(parallel=True)
def run_many_resource_states_jit_br_singledist(Xbig, Zbig, n_CNOT, m, error_pz_all, branching_all):
    """
    Simulate all resource states and samples in one go for branching AND single-emitter distinguishability.
    """
    n_rs = Xbig.shape[0]
    N = Xbig.shape[1]
    n_sample = Xbig.shape[2]
    
    for i in numba.prange(n_rs):
        for s in range(n_sample):
            ep = np.zeros((N, 2), dtype=np.uint8)
            simulate_one_resource_state_jit_br_singledist(ep, error_pz_all[i, s], branching_all[i, s], n_CNOT, m)
            for q in range(N):
                Xbig[i, q, s] = ep[q, 0]
                Zbig[i, q, s] = ep[q, 1]

    return Xbig, Zbig

def optimized_simulate_many_resources_branching_singledist(num_qubits_res_state, error_pz_all, branching_all, m):
    """
    Vectorized simulation over all resource states for branching AND single-emitter distinguishability.
    """
    n_rs = branching_all.shape[0]
    n_sample = branching_all.shape[1]
    N = (num_qubits_res_state * m) + 1  
    n_CNOT = N - 1
    
    Xbig = np.zeros((n_rs, N, n_sample), dtype=np.uint8)
    Zbig = np.zeros((n_rs, N, n_sample), dtype=np.uint8)
    
    run_many_resource_states_jit_br_singledist(Xbig, Zbig, n_CNOT, m, error_pz_all, branching_all)
    
    # Remove spin qubit
    X_error = Xbig[:, 1:, :]
    Z_error = Zbig[:, 1:, :]
    
    return [X_error, Z_error]

#################### Convert physical errors/loss into fusion errors/loss ####################

# RUS functions for sampling logical fusion outcomes, set r=0 to get p_XXZZ_RUS_func
def p_XXZZ_dist_func(eta, N_rep, r, p_fail = 0.5):
    p_e_XX = r/2 - 0.25*r**2 # prob of not getting physical XX outcome, suppose that fusion has already failed
    p_success = (1-p_fail)*( 1 - p_e_XX ) # prob of not getting both physical XX and ZZ fusion outcome 
    p_fail_but_XX = p_fail * (1-p_e_XX)  # prob of failed physical fusion but retrieve XX fusion outcome 
    
    if np.any(N_rep == 0):
        result = 0.0
    else:
        result = eta**2 * p_success + eta**2 * p_success * ( (1- (eta**2*p_fail_but_XX)**N_rep) / (1-eta**2*p_fail_but_XX) - 1)
    
    return result

def p_ZZ_dist_func(eta, N_rep, r, p_fail = 0.5):
    if np.any(N_rep == 0):
        result = 0.0
    else:
        result = 1- (p_XXZZ_dist_func(eta, N_rep, r, p_fail = p_fail) + p_XX_dist_func(eta, N_rep, r, p_fail = p_fail) + 
                     p_erase_dist_func(eta, N_rep, r, p_fail = p_fail)) # if writing in 1-a-b-c this gives floating point error where N_rep=1 it is negative
    
    return result

def p_XX_dist_func(eta, N_rep, r, p_fail = 0.5):
    p_e_XX = r/2 - 0.25*r**2 # prob of not getting physical XX outcome, suppose that fusion has already failed
    p_fail_but_XX = p_fail * (1-p_e_XX)  # prob of failed physical fusion but retrieve XX fusion outcome 
    
    return (p_fail_but_XX*eta**2)**N_rep

def p_erase_dist_func(eta, N_rep, r, p_fail = 0.5):
    p_e_XX = r/2 - 0.25*r**2 # prob of not getting physical XX outcome, suppose that fusion has already failed
    p_fail_but_XX = p_fail * (1-p_e_XX)  # prob of failed physical fusion but retrieve XX fusion outcome 
    p_fail_no_XX = p_fail * (p_e_XX)
    p_success_no_XX = (1-p_fail) * (p_e_XX)
    
    if eta == 1: # consider limiting case for no loss
        final = p_fail_but_XX**(N_rep-1) * p_fail_no_XX
    else:
        a = eta**2 * p_fail_but_XX / (1-eta**2)
        final = (1-eta**2)**(N_rep-1) * (1-a**(N_rep)) / (1-a)  * ( (1-eta**2) + eta**2 * p_fail_no_XX  )
    
    return final

def RUS_ReAttempts(fusions_layer_order, layer_ix, RUS_outcomes_XXZZ_indice_, RUS_lost_outcomes_index, probs_RUS_lost_outcomes_list, qbts_in_fusions,
                   fusion_for_qubits):
    fus_ix_layer = np.where(fusions_layer_order == layer_ix)[0]
    outcomes_current_layer = RUS_outcomes_XXZZ_indice_[list(fus_ix_layer)]
    ZZ_current_layer_indices = list(np.where(outcomes_current_layer == 1)[0]) # check if log fus in current layer has ZZ
    fus_ix_currentZZ = fus_ix_layer[ZZ_current_layer_indices]
    qbts_previous_fus = qbts_in_fusions[fus_ix_currentZZ]-6 # FOR 2nd LAYER FUSION INDEX IS ALWAYS >=6 
    outcome_previous_layer = RUS_outcomes_XXZZ_indice_[fusion_for_qubits[qbts_previous_fus]]
    fus_ix_corrected = fus_ix_currentZZ[list(np.where((outcome_previous_layer[:,0]==1) & (outcome_previous_layer[:,1]==1))[0])]
    RUS_outcomes_XXZZ_indice_[fus_ix_corrected] = np.random.choice(RUS_lost_outcomes_index, len(fus_ix_corrected), p=probs_RUS_lost_outcomes_list)  
    
    return RUS_outcomes_XXZZ_indice_

def RUS_fusion_erasures(num_fusions, qbts_in_fusions, num_qubits_res_state, fusions_primal_isZZ, fusions_layer_order, fusion_for_qubits, p_loss, 
                        p_fusion_success, m_max, noise_type):
    RUS_lost_outcomes_list = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])  # [XX, ZZ] for each case. 0 means no loss in logical XX
    # All 4 cases are [XX (lost in ZZ), ZZ (lost in XX), erasure, XXZZ] for logical fusion
    RUS_lost_outcomes_index = [0, 1, 2, 3]

    if noise_type == 'RUS' or noise_type == 'RUS_reinit':
        probs_RUS_lost_outcomes_list = [p_XX_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success),
                                        p_ZZ_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success),
                                        p_erase_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success),
                                        p_XXZZ_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success)]

        RUS_outcomes_XXZZ_indice_ = np.random.choice(RUS_lost_outcomes_index, num_fusions, p=probs_RUS_lost_outcomes_list)
        
        ## Reinitialisation strategy
        if noise_type == 'RUS_reinit':
            for layer_ix in range(num_qubits_res_state):
                if layer_ix > 0: # only apply to the second layer, the first layer is normal RUS
                    i = 0
                    while i < m_max: # re-attempt RUS for m_max times
                        RUS_outcomes_XXZZ_indice_ = RUS_ReAttempts(fusions_layer_order, layer_ix, RUS_outcomes_XXZZ_indice_, RUS_lost_outcomes_index,
                                                                   probs_RUS_lost_outcomes_list, qbts_in_fusions, fusion_for_qubits)
                        i += 1
        
        RUS_outcomes_XXZZ_indice = list(RUS_outcomes_XXZZ_indice_)

    else:
        raise ValueError('Only implemented for ''[\'RUS\']')
    
    lost_fusion_XXZZ = RUS_lost_outcomes_list[RUS_outcomes_XXZZ_indice]
    lost_fusions_primal = [lost_fusion_XXZZ[i][j] for i,j in enumerate(fusions_primal_isZZ)]
    lost_fusions_dual = [lost_fusion_XXZZ[i][j] for i,j in enumerate(1-fusions_primal_isZZ)]

    
    primal_errors = None
    dual_errors = None
    
    return primal_errors, dual_errors, lost_fusions_primal, lost_fusions_dual

# For each logical fusion, compute the logical fusion error
@numba.njit
def compute_fus_errors_jit(num_fusions, m_max, qbts_in_fusions, RUS_outcomes, all_XXZZ_indices, fusions_primal_isZZ, qubit_errors, fus_error_list):
    for log_fus_ix in range(num_fusions):
        # Get indices for the two logical qubits involved in the fusion
        q1 = qbts_in_fusions[log_fus_ix, 0]
        q2 = qbts_in_fusions[log_fus_ix, 1]
        outcome = RUS_outcomes[log_fus_ix]
        
        if outcome == 3:
            attempt = all_XXZZ_indices[log_fus_ix]
            sum1 = 0
            for k in range(attempt):
                # Sum the errors on the second (logical Z) component modulo 2
                sum1 += ( (qubit_errors[q1, k, 1] + qubit_errors[q2, k, 1]) & 1 )
            error_val_XX = sum1 & 1
            fus_error_list[fusions_primal_isZZ[log_fus_ix], log_fus_ix] = error_val_XX
            # For the dual (logical ZZ) outcome, take the error from the first (logical X) component at the final attempt
            e1 = (qubit_errors[q1, attempt-1, 0] + qubit_errors[q2, attempt-1, 0]) & 1
            fus_error_list[1 - fusions_primal_isZZ[log_fus_ix], log_fus_ix] = e1
        
        elif outcome == 0:
            sum1 = 0
            for k in range(m_max):
                sum1 += ((qubit_errors[q1, k, 1] + qubit_errors[q2, k, 1]) & 1)
            error_val_XX = sum1 & 1
            fus_error_list[fusions_primal_isZZ[log_fus_ix], log_fus_ix] = error_val_XX
    
    return fus_error_list

def fusion_ErasureError( num_fusions, qbts_in_fusions, num_qubits_res_state, fusions_primal_isZZ, qubit_errors, fusions_layer_order, fusion_for_qubits, 
                        p_loss, p_err, dist_err, p_fusion_success, p_fusion_error_ZZ, m_max, noise_type, blinked_fusions, loss_mask):

    fusion_loss = np.zeros((num_fusions, 2), dtype=bool)
    for fus_ix, (q1, q2) in enumerate(qbts_in_fusions):
        # if *any* attempt on either qubit was erased, mark fusion lost in both XX & ZZ
        if loss_mask[q1].any() or loss_mask[q2].any():
            fusion_loss[fus_ix] = [True, True]    

    RUS_lost_outcomes_list = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])  # logical [XX, ZZ] for each case. [0, 1] means no loss in XX, loss in ZZ
    " All 4 cases are [XX (lost in ZZ), ZZ (lost in XX), erasure, XXZZ] "
    RUS_lost_outcomes_index = [0, 1, 2, 3]

    if noise_type in ('dist_RUS', 'dist_RUS_reinit'):
        # Here p_err = r (Reflectivity of beamsplitter)
        probs_RUS_lost_outcomes_list = [p_XX_dist_func(1-p_loss, m_max, p_err, p_fail=1-p_fusion_success),
                                        p_ZZ_dist_func(1-p_loss, m_max, p_err, p_fail=1-p_fusion_success),
                                        p_erase_dist_func(1-p_loss, m_max, p_err, p_fail=1-p_fusion_success),
                                        p_XXZZ_dist_func(1-p_loss, m_max, p_err, p_fail=1-p_fusion_success)]
        
        RUS_outcomes_XXZZ_indice_ = np.random.choice(RUS_lost_outcomes_index, num_fusions, p=probs_RUS_lost_outcomes_list)

        # Reinitialisation loop if requested
        if noise_type in ('dist_RUS_reinit'):
            for layer_ix in range(num_qubits_res_state):
                if layer_ix > 0:  # skip first layer
                    for _ in range(m_max):
                        RUS_outcomes_XXZZ_indice_ = RUS_ReAttempts(fusions_layer_order, layer_ix, RUS_outcomes_XXZZ_indice_, RUS_lost_outcomes_index,
                                                                   probs_RUS_lost_outcomes_list,
                                                                   qbts_in_fusions, fusion_for_qubits)

        RUS_outcomes_XXZZ_indice = list(RUS_outcomes_XXZZ_indice_)
        lost_fusion_XXZZ = RUS_lost_outcomes_list[RUS_outcomes_XXZZ_indice]
        lost_fusion_XXZZ |= fusion_loss

        # split out primal/dual erasures
        primal_lost_fusions = [lost_fusion_XXZZ[i][j]
            for i, j in enumerate(fusions_primal_isZZ)]
        
        dual_lost_fusions = [lost_fusion_XXZZ[i][j]
            for i, j in enumerate(1 - fusions_primal_isZZ)]

        # sample any remaining ZZ-errors on alive fusions
        fus_error_list = np.zeros((2, num_fusions), dtype=np.int8)
        for log_fus_ix in range(num_fusions):
            if RUS_outcomes_XXZZ_indice[log_fus_ix] in (1, 3):
                fus_error_list[1 - fusions_primal_isZZ[log_fus_ix], log_fus_ix] = np.random.binomial(1, p_fusion_error_ZZ)

        primal_errors = fus_error_list[0][np.where(~np.array(primal_lost_fusions))[0]]
        dual_errors = fus_error_list[1][np.where(~np.array(dual_lost_fusions))[0]]

    elif noise_type in ('Spin_X_RUS', 'Spin_Z_RUS', 'Spin_depol_RUS', 'RUS_branching', 'RUS_blinking', 'RUS_blinking_reinit', 'single_emitter_dist_RUS'):
        # 1) Compute pure-RUS outcome probabilities
        probs = [p_XX_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success),
                 p_ZZ_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success),
                 p_erase_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success),
                 p_XXZZ_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success)]
        
        RUS_outcomes_XXZZ_indice_ = np.random.choice(RUS_lost_outcomes_index, num_fusions, p=probs)

        # 2) Immediately fold blinking losses into the RUS indices
        #    so that re-init can re-draw even blink-caused losses.
        # Map indices → boolean loss pairs:
        lost0 = RUS_lost_outcomes_list[RUS_outcomes_XXZZ_indice_]
        lost0 |= blinked_fusions                                   
        lost0 |= fusion_loss
        
        # Build reverse map: (xx,zz) → index
        idx_map = { tuple(pair): idx for idx, pair in enumerate(RUS_lost_outcomes_list) }
        # Remap back to RUS indices
        RUS_outcomes_XXZZ_indice_ = np.array([ idx_map[tuple(pair)] for pair in lost0])

        if noise_type == 'RUS_blinking_reinit':
            for layer_ix in range(num_qubits_res_state):
                if layer_ix > 0:
                    for _ in range(m_max):
                        RUS_outcomes_XXZZ_indice_ = RUS_ReAttempts(
                            fusions_layer_order,
                            layer_ix,
                            RUS_outcomes_XXZZ_indice_,
                            RUS_lost_outcomes_index,
                            probs,
                            qbts_in_fusions,
                            fusion_for_qubits)

        # 4) Final lost‐fusion mask is just whatever re-init left us
        RUS_outcomes_XXZZ_indice = list(RUS_outcomes_XXZZ_indice_)
        lost_fusion_XXZZ = RUS_lost_outcomes_list[RUS_outcomes_XXZZ_indice]

        # Determine primal/dual erasure lists
        primal_lost_fusions = [lost_fusion_XXZZ[i][j] for i, j in enumerate(fusions_primal_isZZ)]
        dual_lost_fusions   = [lost_fusion_XXZZ[i][j] for i, j in enumerate(1 - fusions_primal_isZZ)]

        # Initialize fusion error array
        fus_error_list = np.zeros((2, num_fusions), dtype=np.int8)

        # Probabilities for achieving XXZZ in each attempt
        probs_XXZZ_attempt_list = np.array([p_fusion_success * (1-p_fusion_success)**i for i in range(m_max)])
        
        attempt_index = [i+1 for i in range(m_max)]
        all_XXZZ_indices = np.random.choice(attempt_index, num_fusions, p=probs_XXZZ_attempt_list / np.sum(probs_XXZZ_attempt_list))

        # Compute logical fusion error bits via JIT helper
        RUS_outcomes_array       = np.array(RUS_outcomes_XXZZ_indice, dtype=np.int64)
        qbts_in_fusions_arr      = np.array(qbts_in_fusions, dtype=np.int64)
        fusions_primal_isZZ_arr  = np.array(fusions_primal_isZZ, dtype=np.int64)

        fus_error_list = compute_fus_errors_jit(num_fusions, m_max, qbts_in_fusions_arr, RUS_outcomes_array, all_XXZZ_indices, fusions_primal_isZZ_arr, 
                                                qubit_errors, fus_error_list)

        # Extract logical errors for fusions that were not lost
        primal_errors = fus_error_list[0][np.where(~np.array(primal_lost_fusions))[0]]
        dual_errors   = fus_error_list[1][np.where(~np.array(dual_lost_fusions))[0]]

    elif noise_type == 'RUS_branching_singledist':
        # Compute probabilities for the different RUS lost-outcome types.
        probs_RUS_lost_outcomes_list = [p_XX_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success), 
                                        p_ZZ_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success), 
                                        p_erase_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success), 
                                        p_XXZZ_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success)] # p_err = 0 means r = 0 for indistinguishable photons
        
        RUS_outcomes_XXZZ_indice = list(np.random.choice(RUS_lost_outcomes_index, num_fusions, p=probs_RUS_lost_outcomes_list))
        lost_fusion_XXZZ = RUS_lost_outcomes_list[RUS_outcomes_XXZZ_indice]  
        # fusions_primal_isZZ: for each logical fusion, 0=XX and 1=ZZ
        
        lost_fusion_XXZZ = lost_fusion_XXZZ | blinked_fusions
        lost_fusion_XXZZ |= fusion_loss
    
        # Determine which fusions are lost in the primal and dual outcomes
        primal_lost_fusions = [lost_fusion_XXZZ[i][j] for i, j in enumerate(fusions_primal_isZZ)]
        dual_lost_fusions   = [lost_fusion_XXZZ[i][j] for i, j in enumerate(1 - fusions_primal_isZZ)]
    
        # Initialize fusion error array
        fus_error_list = np.zeros((2, num_fusions), dtype=np.int8)
    
        # Probabilities for achieving a fusion (XXZZ outcome) in different numbers of attempts.
        probs_XXZZ_attempt_list = np.array([p_fusion_success*(1-p_fusion_success)**i for i in range(m_max)])
        attempt_index = [i+1 for i in range(m_max)]
        all_XXZZ_indices = np.random.choice(attempt_index, num_fusions, p=probs_XXZZ_attempt_list / np.sum(probs_XXZZ_attempt_list))
        RUS_outcomes_array   = np.array(RUS_outcomes_XXZZ_indice, dtype=np.int64)
        qbts_in_fusions_arr  = np.array(qbts_in_fusions, dtype=np.int64)
        fusions_primal_isZZ_arr = np.array(fusions_primal_isZZ, dtype=np.int64)
        
        fus_error_list = compute_fus_errors_jit(num_fusions, m_max, qbts_in_fusions_arr, RUS_outcomes_array, all_XXZZ_indices, fusions_primal_isZZ_arr, 
                                                qubit_errors, fus_error_list)
    
        # Extract logical errors for fusions that were not lost.
        primal_errors = fus_error_list[0][np.where(np.logical_not(primal_lost_fusions))[0]]
        dual_errors   = fus_error_list[1][np.where(np.logical_not(dual_lost_fusions))[0]]
        
    elif noise_type == 'RUS_dist_SpinDepol':
        # Here, dist_err = r (Reflectivity of beamsplitter) 
        probs_RUS_lost_outcomes_list = [p_XX_dist_func(1-p_loss, m_max, dist_err, p_fail=1-p_fusion_success),
                                        p_ZZ_dist_func(1-p_loss, m_max, dist_err, p_fail=1-p_fusion_success),
                                        p_erase_dist_func(1-p_loss, m_max, dist_err, p_fail=1-p_fusion_success),
                                        p_XXZZ_dist_func(1-p_loss, m_max, dist_err, p_fail=1-p_fusion_success)]
        
        RUS_outcomes_XXZZ_indice_ = np.random.choice(RUS_lost_outcomes_index, num_fusions, p=probs_RUS_lost_outcomes_list)
        
        # Reinitialisation strategy
        for layer_ix in range(num_qubits_res_state):
            if layer_ix > 0: # only apply to the second layer, the first layer is normal RUS
                i = 0
                while i < m_max: # re-attempt RUS for m_max times
                    RUS_outcomes_XXZZ_indice_ = RUS_ReAttempts(fusions_layer_order, layer_ix, RUS_outcomes_XXZZ_indice_, RUS_lost_outcomes_index,
                                                               probs_RUS_lost_outcomes_list, qbts_in_fusions, fusion_for_qubits)
                    i += 1
                    
        RUS_outcomes_XXZZ_indice = list(RUS_outcomes_XXZZ_indice_)
        
        lost_fusion_XXZZ = RUS_lost_outcomes_list[RUS_outcomes_XXZZ_indice] # dimension: number of fusions x 2
        "fusions_primal_isZZ: in primal fusion syndrome lattice, for each logical fusion, which type 0=XX, 1=ZZ it belongs to."  
        "fusions_primal_isZZ[fus_ix] = 0 means fus_ix is logical XX in primal. 1 means fus_ix is logical ZZ in primal"
        lost_fusion_XXZZ |= fusion_loss

        # list of fusion erasure in primal outcome
        primal_lost_fusions = [lost_fusion_XXZZ[i][j] for i,j in enumerate(fusions_primal_isZZ)] 
        dual_lost_fusions = [lost_fusion_XXZZ[i][j] for i,j in enumerate(1-fusions_primal_isZZ)]

        "Sample fusion errors for logical XX = XXXXXX, logical ZZ = ZZ"
        fus_error_list = np.zeros(shape = (2,num_fusions),dtype=np.int8) # list of logical fusion syndromes

        # Probabilities for getting logical XXZZ in different numbers of attempts
        probs_XXZZ_attempt_list = np.array([p_XXZZ_dist_func(1-p_loss, i+1, dist_err, p_fail=1-p_fusion_success) -
                                            p_XXZZ_dist_func(1-p_loss, i, dist_err, p_fail=1-p_fusion_success) for i in range(m_max)])
        
        attempt_index = [i+1 for i in range(m_max)]
        all_XXZZ_indices = np.random.choice(attempt_index, num_fusions, p=probs_XXZZ_attempt_list / np.sum(probs_XXZZ_attempt_list))
        # Probabilities for getting logical ZZ in different numbers of attempts
        probs_ZZ_attempt_list = np.array([p_ZZ_dist_func(1-p_loss, i+1, dist_err, p_fail=1-p_fusion_success) -
                                          p_ZZ_dist_func(1-p_loss, i, dist_err, p_fail=1-p_fusion_success) for i in range(m_max)])
        
        if not np.any(probs_ZZ_attempt_list): 
            all_ZZ_indices = None 
        else:
            ZZ_attempt_index = [i+1 for i in range(m_max)]
            all_ZZ_indices = np.random.choice(ZZ_attempt_index, num_fusions, p=probs_ZZ_attempt_list / np.sum(probs_ZZ_attempt_list))
        
        for log_fus_ix in range(num_fusions):
            logqbt1_ix, logqbt2_ix = qbts_in_fusions[log_fus_ix] # logqbt1_ix contains m=N_rep number of qubits (max attempts)

            if RUS_outcomes_XXZZ_indice[log_fus_ix] == 3: # if log fusion ends up with logical XXZZ
            # consider spin error in logical XX
                fus_error_list[fusions_primal_isZZ[log_fus_ix]][log_fus_ix] = np.remainder(
                    np.sum(qubit_errors[logqbt1_ix][:all_XXZZ_indices[log_fus_ix]] + 
                           qubit_errors[logqbt2_ix][:all_XXZZ_indices[log_fus_ix]] % 2,axis=0), 2)[1] # odd number of Z leads to error in logical XX = XX^n where n is the number of attempts. No XX physical fusion failure.
            
                # consider spin error in logical ZZ
                fus_error_list[1-fusions_primal_isZZ[log_fus_ix]][log_fus_ix] = (((qubit_errors[logqbt1_ix][all_XXZZ_indices[log_fus_ix]-1] +
                                                                                    qubit_errors[logqbt2_ix][all_XXZZ_indices[log_fus_ix]-1]) % 2)[0] +
                                                                                    np.random.binomial(1,p_fusion_error_ZZ) ) % 2 
            # (1) spin error: list of ZZ physical fusion errors, odd number of X that gives error to physical ZZ fusion, which leads to logical ZZ
            # (2) dist error: sample fusion error on the only ZZ outcome. For RUS we always stop at getting one ZZ physical outcome
                
            elif RUS_outcomes_XXZZ_indice[log_fus_ix] == 1: # if log fusion ends up with logical ZZ
                fus_error_list[1-fusions_primal_isZZ[log_fus_ix]][log_fus_ix] = (((qubit_errors[logqbt1_ix][all_ZZ_indices[log_fus_ix]-1] +
                                                                                   qubit_errors[logqbt2_ix][all_ZZ_indices[log_fus_ix]-1]) % 2)[0] + 
                                                                                   np.random.binomial(1,p_fusion_error_ZZ) ) % 2
            # (1) spin error: list of ZZ physical fusion errors, odd number of X that gives error to physical ZZ fusion, which leads to logical ZZ
            # (2) dist error: sample fusion error on the only ZZ outcome. For RUS we always stop at getting one ZZ physical outcome

            elif RUS_outcomes_XXZZ_indice[log_fus_ix] == 0: # if log fusion ends up with logical XX only
            # consider spin error in logical XX
                fus_error_list[fusions_primal_isZZ[log_fus_ix]][log_fus_ix] = np.remainder(
                    np.sum(qubit_errors[logqbt1_ix]+ qubit_errors[logqbt2_ix] % 2,axis=0), 2)[1] # odd number of Z leads to error in logical XX = XXXXXX. No XX physical fusion failure.
                
                # we don't know at what values of N_rep exactly fusion is successful or biased to give ZZ outcome
                # so we sample the error given that we received a logical ZZ outcome
                # if fusions_primal_isZZ[log_fus_ix] = 0, then log_fus_ix is logical ZZ fusion in dual (fus_error_list[1]), or XX in primal
                # Distinguishability does not lead to physical XX error so does not affect logical XX, only affect physical ZZ
                
        primal_errors = fus_error_list[0][np.where(np.logical_not(primal_lost_fusions))[0]]
        dual_errors = fus_error_list[1][np.where(np.logical_not(dual_lost_fusions))[0]]

    else:
        primal_errors = None
        dual_errors = None
        probs_RUS_lost_outcomes_list = [p_XX_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success),
                                        p_ZZ_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success),
                                        p_erase_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success),
                                        p_XXZZ_dist_func(1-p_loss, m_max, 0, p_fail=1-p_fusion_success)] 
        
        RUS_outcomes_XXZZ_indice = list(np.random.choice(RUS_lost_outcomes_index, num_fusions, p=probs_RUS_lost_outcomes_list))
        lost_fusion_XXZZ = RUS_lost_outcomes_list[RUS_outcomes_XXZZ_indice]
        primal_lost_fusions = [lost_fusion_XXZZ[i][j] for i,j in enumerate(fusions_primal_isZZ)] #
        dual_lost_fusions = [lost_fusion_XXZZ[i][j] for i,j in enumerate(1-fusions_primal_isZZ)]
            
    return primal_errors, dual_errors, primal_lost_fusions, dual_lost_fusions

if __name__ == '__main__':
    from timeit import default_timer
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    # Example simulation for photon loss

    p_fusion_fail = 0.5
    num_trials = 10000
    L_list = np.array([4, 6, 8])
    A = 1
    D = 0
    m_range = [4] # Number of photons encoding each qubit
    loss = np.linspace(0, 0.01, 15)
    err_vs_eras_vals = np.array([(0, l, 0, 0) for l in loss]) # [error, loss, distinguishability, branching]

    for m in m_range:
        start_t = default_timer()
        print(f'Running for m={m}')
        
        Chain_data = []
        for L_ix, L in enumerate(L_list):
            print('   Doing L =', L)
            this_data = decoder_successprob_error_vs_loss_list_parallelized_encoded(
                err_vs_eras_vals, p_fusion_fail, L, m, A, D, num_loss_trials=num_trials, num_ec_runs_per_loss_trial=1,
                noise_mechanism='dist_RUS_reinit', decoding_weights='None')
            Chain_data.append(this_data)
        
        end_t = default_timer()
        print(f'Completed m={m} in {end_t - start_t:.2f} s')
        
        log_err_rate = Chain_data
        figure_name = f'Chain_m{m}_data_50ff'
        
        cm = plt.get_cmap('plasma')
        cNorm = colors.Normalize(vmin=0, vmax=1.2)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        
        plots_colors = [scalarMap.to_rgba(x / (len(L_list) - 1)) for x in range(len(L_list))]
        fig = plt.figure()
        
        for L_ix, L in enumerate(L_list):
            plt.errorbar(
                loss * 100, log_err_rate[L_ix],
                yerr=(log_err_rate[L_ix] * (1 - log_err_rate[L_ix]) / num_trials) ** 0.5,
                label=r"$L=${}".format(L),
                color=plots_colors[L_ix])
        
        plt.yscale('log')
        plt.xlabel("Loss rate (%)")
        plt.ylabel("Logical error rate")
        plt.title(f'm = {m}')
        plt.minorticks_on()
        plt.legend(loc='lower right')
        plt.grid(True, which="both", color='0.9')
        fig.tight_layout()
        plt.show()
