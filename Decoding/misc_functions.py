import numpy as np
from itertools import combinations
from time import sleep



#################### Functions used to handle the H matrices and degeneracies, and associate probabilities weights for the decoder ####################

def merge_multiedges_in_Hmat(Hmat):
    uniq_H, new_ixs, inverse_ixs, occ_counts = np.unique(Hmat, axis=1, return_index=True, return_inverse=True,
                                                         return_counts=True)
    # Check if there is a column with all zeros (qbt in no syndrome) and removes it
    if np.any(uniq_H[:, 0]):
        return uniq_H, new_ixs, inverse_ixs, occ_counts, True
    else:
        return uniq_H[:, 1:], new_ixs[1:], inverse_ixs - 1, occ_counts[1:], False

def merge_multiedges_in_Hmat_faster(qbt_syndr_mat):
    uniq_qbt_syndr_mat, new_ixs, inverse_ixs, occ_counts = np.unique(qbt_syndr_mat, axis=0, return_index=True,
                                                                     return_inverse=True, return_counts=True)

    first_good_index = np.argmax(uniq_qbt_syndr_mat.T[0] >= 0)
    new_inverse_ixs = np.array([i for i in inverse_ixs - first_good_index if i >= 0]) # added equal sign to i > 0, 19 Jun 2024

    return uniq_qbt_syndr_mat[first_good_index:], new_ixs[
                                                  first_good_index:], new_inverse_ixs, occ_counts[first_good_index:], False

def get_multiedge_errorprob(error_type='iid', **kwargs):
    if error_type == 'iid':
        if 'p' and 'occ_counts' in kwargs:
            p = kwargs['p']
            return np.vectorize(lambda x: 0.5 * (1 - ((1 - 2. * p) ** x)))(kwargs['occ_counts'])
        else:
            raise ValueError(
                'Required input for "iid" error type not provided. Required inputs are: \n - Error rate "p"\n - Numpy '
                'array "occ_counts" containing list of degeneracies for each multiedge.')
    
    elif error_type == 'weight-iid':
        if 'p' and 'valencies_list' and 'inv_mat' and 'num_good_qbts' in kwargs:
            # print('\n\nIn get_multiedge_errorprob function')
            p = kwargs['p']
            num_good_qbts = kwargs['num_good_qbts']
            valences_multiedges = [[] for _ in range(num_good_qbts)]
            valencies_list = kwargs['valencies_list']

            # print('\nnum_good_qbts:')
            # print(num_good_qbts)
            # print('\nvalencies_list shape:', valencies_list.shape)
            # print('valences_multiedges shape:', len(valences_multiedges))
            # print('inv_mat:', kwargs['inv_mat'])

            # start_t = default_timer()
            for qbt_ix, good_qbt_ix in enumerate(kwargs['inv_mat']):
                # print('in inv_mat ix', qbt_ix, ' (qbt_ix) there is new H col', good_qbt_ix, '(good_qbt_ix)')
                if good_qbt_ix > 0:
                    valences_multiedges[good_qbt_ix].append(valencies_list[qbt_ix])
            
            multi_edge_err_probs = np.zeros(num_good_qbts, dtype=np.dtype('float64'))
            for multiedge_ix, vals in enumerate(valences_multiedges):
                uniq_vals, counts = np.unique(vals, return_counts=True)
                no_loss_prob = np.prod([(1 - p * uniq_vals[ix]) ** c for ix, c in enumerate(counts)])
                multi_edge_err_probs[multiedge_ix] = sum(
                    [no_loss_prob * c * p * uniq_vals[ix] / (1 - p * uniq_vals[ix]) for ix, c in enumerate(counts)])

            return multi_edge_err_probs

        else:
            raise ValueError(
                'Required input for "weight-iid" error type not provided. Required inputs are: \n - Error rate "p"\n - Numpy '
                'array "valencies_list" containing list of the valency for each qubit.\n '
                '- Numpy array "inv_mat" cointining to which multiedge each qubit contributes to\n'
                ' - int "num_good_qbts" the number of multiedges')
    else:
        raise ValueError('Only accepted error types are: ["iid", "weight-iid"]')


def get_Hmat_weights(error_type='iid', bound_for_zero=100., **kwargs): # add . after 100 so that the error probs are not integers, 19 Jun 2024
    if error_type in ['iid', 'weight-iid']:
        if 'multiedge_error_probs' in kwargs:
            return np.vectorize(lambda x: bound_for_zero if np.isclose(x, 0) else np.log(1 - x) - np.log(x))(
                kwargs['multiedge_error_probs'])
        else:
            raise ValueError(
                'Required input for "' + error_type + '" error type not provided. Required inputs are: \n - Numpy array '
                                                      '"multiedge_error_probs" with error probabilities for each edge of the matching graph.')
    else:
        raise ValueError('Only accepted error types are: ["iid"]')
        
def get_paired_qbt_list(qbt_syndr_mat):
    uniq_qbt_syndr_mat, new_ixs, inverse_ixs, occ_counts = np.unique(qbt_syndr_mat, axis=0,
                                                                     return_index=True,
                                                                     return_inverse=True, return_counts=True)

    all_pairs = np.array([np.where(inverse_ixs == ix)[0] for ix in range(len(new_ixs))], dtype=object) 

    for pair in all_pairs:
        if len(pair) == 2:
            paired_qbt_list[pair[0]] = pair[1]
            paired_qbt_list[pair[1]] = pair[0]
        elif len(pair) == 1:
            paired_qbt_list[0] = pair[0]
        else:
            raise ValueError('Pairs must have 2 or 1 elements')
            
    return paired_qbt_list

def fusion_error_merged(new_ixs,inverse_ixs,fusion_errors):
    """
    fusion_error: list of fusion errors on either primal or dual
    Idea: convert fusion error/syndrome list (before merging edges) to fusion syndrome (after merging edges) 
    """
    new_fusion_errors = np.zeros(len(new_ixs), dtype=np.uint8)
    for ix in range(len(new_ixs)):
        new_ix = np.where(inverse_ixs == ix)[0]
        new_fusion_errors[ix] = np.bitwise_xor.reduce(fusion_errors[new_ix])
    return new_fusion_errors
