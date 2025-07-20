import numpy as np
import ctypes
import platform  ## Used in the c++ wrappers testing on which operating system we are working on
import os

from time import sleep

cwd = os.getcwd()

#########################################################################################################
###### Wrapper functions to make Noise Sampling.
###### Sampling operations performed in Cpp

os_system = platform.system()

if os_system == 'Linux':
    try:
        NScpp_header = ctypes.cdll.LoadLibrary('./libNoiseSampling.so')
    except:
        NScpp_header = ctypes.cdll.LoadLibrary(os.path.join(cwd, 'FusionLatticesAnalysis', 'libNoiseSampling.so'))
    print('Loaded C++ noise sampling functions for Linux OS')
# elif os_system == 'Windows':
# try:
#     LTcpp_header = ctypes.cdll.LoadLibrary('./libLossDec_win.dll')
# except:
#     LTcpp_header = ctypes.cdll.LoadLibrary(os.path.join(cwd, 'FusionLatticesAnalysis', 'libLossDec_win.dll'))
# print('Loaded C++ linear algebra functions for Windows OS')
else:
    raise ValueError('Os system not supported: only Linux')
NScpp_header.SampleFusionLoss_passive.argtypes = [ctypes.c_int, ctypes.c_int,
                                                  ctypes.POINTER(ctypes.c_int),
                                                  ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool),
                                                  ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
NScpp_header.SampleFusionLoss_simpleactive.argtypes = [ctypes.c_int, ctypes.c_int,
                                                       ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                                       ctypes.POINTER(ctypes.c_int),
                                                       ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool),
                                                       ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_bool),
                                                       ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]

NScpp_header.SampleFusionErrors_uncorrelated.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                                         ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                                         ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool),
                                                         ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool),
                                                         ctypes.c_float, ctypes.c_float]

NScpp_header.multiedge_errorprob_uncorrelated.argtypes = [ctypes.c_int,
                                                          ctypes.c_int,
                                                          ctypes.POINTER(ctypes.c_int),
                                                          ctypes.c_bool,
                                                          ctypes.POINTER(ctypes.c_float),
                                                          ctypes.POINTER(ctypes.c_bool),
                                                          ctypes.POINTER(ctypes.c_bool),
                                                          ctypes.c_float, ctypes.c_float]

NScpp_header.multiedge_errorprob_uncorrelated_precharnoise.argtypes = [ctypes.c_int,
                                                          ctypes.c_int,
                                                          ctypes.POINTER(ctypes.c_int),
                                                          ctypes.POINTER(ctypes.c_float),
                                                          ctypes.POINTER(ctypes.c_float),]


NScpp_header.propagate_errors_multiedge.argtypes = [ctypes.c_int,
                                                    ctypes.c_int,
                                                    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                                    ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool),
                                                    ctypes.POINTER(ctypes.c_bool)]


def sample_fusion_loss_passive(num_qubits, num_fusions, qbts_in_fusions, fusion_biases, p_loss, p_fail,
                               p_fail_biased=0., fusion_is_physical=True):
    lost_fusions = np.zeros((num_fusions, 2), dtype=np.uint8)

    NScpp_header.SampleFusionLoss_passive(num_qubits, num_fusions,
                                          qbts_in_fusions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                          fusion_biases.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                          lost_fusions.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                          p_loss, p_fail, p_fail_biased, fusion_is_physical)

    return lost_fusions


def sample_fusion_loss_activesimple(num_qubits, num_fusions, qbts_in_fusions, fusions_timeorder, time_of_fusions,
                                    fusion_biases, paired_fusions, p_loss, p_fail, p_fail_biased=0., fusion_is_physical=True):
    lost_fusions = np.zeros((num_fusions, 2), dtype=np.uint8)
    single_pauli_meas_list = np.zeros(num_fusions, dtype=np.uint8)
    REF_fusion_biases = fusion_biases.copy()

    NScpp_header.SampleFusionLoss_simpleactive(num_qubits, num_fusions,
                                               qbts_in_fusions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                               fusions_timeorder.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                               time_of_fusions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                               REF_fusion_biases.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                               single_pauli_meas_list.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                               paired_fusions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                               lost_fusions.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                               p_loss, p_fail, p_fail_biased, fusion_is_physical)

    return lost_fusions, REF_fusion_biases, single_pauli_meas_list


def sample_fusion_errors_uncorrelated(num_fusions, alive_primal_fusions, alive_dual_fusions,
                                      fusion_biases, single_pauli_meas_list, p_err_biased, p_err_unbiased=None):
    if p_err_unbiased == None:
        p_err_unbiased = p_err_biased

    primal_errors = np.zeros(num_fusions, dtype=np.uint8)
    dual_errors = np.zeros(num_fusions, dtype=np.uint8)

    num_alive_primal_fusions = len(alive_primal_fusions)
    num_alive_dual_fusions = len(alive_dual_fusions)

    NScpp_header.SampleFusionErrors_uncorrelated(num_alive_primal_fusions,
                                                 alive_primal_fusions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                 num_alive_dual_fusions,
                                                 alive_dual_fusions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                 primal_errors.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                                 dual_errors.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                                 fusion_biases.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                                 single_pauli_meas_list.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                                 p_err_biased, p_err_unbiased)

    return primal_errors, dual_errors


def multiedge_errorprob_uncorrelated(num_fusions, num_multiedges, multiedge_of_fusions, is_primal, fusion_biases,
                                     single_pauli_meas_list, p_err_biased, p_err_unbiased):
    multiedge_errorprobs = np.zeros(num_multiedges, dtype=ctypes.c_float)

    # print(multiedge_of_fusions)
    # print(multiedge_errorprobs)
    # print(fusion_biases)

    NScpp_header.multiedge_errorprob_uncorrelated(num_fusions,
                                                  num_multiedges,
                                                  multiedge_of_fusions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                  is_primal,
                                                  multiedge_errorprobs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                  fusion_biases.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                                  single_pauli_meas_list.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                                  p_err_biased, p_err_unbiased)

    return multiedge_errorprobs


def multiedge_errorprob_uncorrelated_precharnoise(num_fusions, num_multiedges, multiedge_of_fusions, p_err_list):
    multiedge_errorprobs = np.zeros(num_multiedges, dtype=ctypes.c_float)

    # print(multiedge_of_fusions)
    # print(multiedge_errorprobs)
    # print(fusion_biases)

    NScpp_header.multiedge_errorprob_uncorrelated_precharnoise(num_fusions,
                                                  num_multiedges,
                                                  multiedge_of_fusions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                  multiedge_errorprobs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                  p_err_list.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    return multiedge_errorprobs


def propagate_errors_multiedge(num_syndromes, num_multiedges, syndromes_of_multiedge, multiedge_of_fusions,
                               errors_in_fusions):
    fired_syndromes = np.zeros(num_syndromes, dtype=np.uint8)
    errors_in_multiedges = np.zeros(num_multiedges, dtype=np.uint8)

    # print('syndromes_of_multiedge\n', syndromes_of_multiedge)
    # print('  num_multiedges:', num_multiedges)
    # print('  num_syndromes:', num_syndromes)
    # print('  num_fusions:', len(errors_in_fusions))
    # sleep(1)

    NScpp_header.propagate_errors_multiedge(errors_in_fusions.shape[0],
                                            num_multiedges,
                                            syndromes_of_multiedge.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                            multiedge_of_fusions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                            errors_in_fusions.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                            fired_syndromes.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                            errors_in_multiedges.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)))

    return fired_syndromes, errors_in_multiedges
