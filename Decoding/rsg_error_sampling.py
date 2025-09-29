import numpy as np

class ErrorSampler:
    def __init__(self, hadamards, x_flip_rate=0.01, y_flip_rate=0.01, z_flip_rate=0.01, good_q_ixs=None, hadamard_photons=True):
        """

        :param hadamards:
        :param x_flip_rate:
        :param z_flip_rate:
        """
        self.hadamards = hadamards
        self.nq = len(self.hadamards) + 1
        self.ex = x_flip_rate
        self.ey = y_flip_rate
        self.ez = z_flip_rate
        self.ts_errors = None
        self.good_q_ixs = good_q_ixs

        self.photon_hadamards = [ix for ix, h in enumerate(hadamards) if not h]
        # for GHZ state, use self.photon_hadamards = list(range(len(hadamards) + 1))
        self.error_maps = dict(zip(['xx', 'xz', 'zx', 'zz'], self.construct_ts_rs_map(apply_photon_hadamards=hadamard_photons)))


    def sample_ts_errors_xyz(self, px, py, pz, num_ts, num_samples):
        """
        Sample errors in timesteps for an error model in which x,y,z errors occur on the spin with probabilities px, py, pz
        :param px: Probability of X error per time step
        :param py: Probability of Y error per time step
        :param pz: Probability of Z error per time step
        :param num_ts: Number of time steps (equal to number of qubits + 1)
        :param num_samples: Number of shots to sample
        :return: x_flips, z_flips. Binary matrices indicating where X and Z stabilizers on the spin are flipped.
        Note that the returned quantity is the stabilizers that are flipped and NOT the errors
        """
        assert (px + py + pz <= 1)
        ts_errors_all = np.random.choice((0, 1, 2, 3), p=(1 - px - py - pz, px, py, pz),
                                         size=(num_ts, num_samples))
        x_flips = (ts_errors_all == 1) + (ts_errors_all == 2)
        z_flips = (ts_errors_all > 1)
        return x_flips.astype(np.float32), z_flips.astype(np.float32)

    def construct_ts_rs_map(self, apply_photon_hadamards=True):
        """
        ab_map maps our physical a errors on the spin to physical b errors on the photons
        :return:
        """
        xx_map = np.zeros(shape=(self.nq, self.nq + 1), dtype=np.float32)
        zz_map = np.zeros(shape=(self.nq, self.nq + 1), dtype=np.float32)
        xz_map = np.zeros(shape=(self.nq, self.nq + 1), dtype=np.float32)
        zx_map = np.zeros(shape=(self.nq, self.nq + 1), dtype=np.float32)
        for loc in range(self.nq + 1):
            if loc == 0:
                zz_map[0, loc] = 1
            elif loc < self.nq:
                zz_map[loc, loc] = 1
                xx_map[loc, loc] = 1
                for h_loc in range(loc, self.nq - 1):
                    hadamard = self.hadamards[h_loc]
                    if hadamard:
                        xz_map[h_loc + 1, loc] = 1
                        break
                    else:
                        xx_map[h_loc + 1, loc] = 1
            else:
                zz_map[self.nq - 1, loc] = 1
        if apply_photon_hadamards:
            # Apply photon hadamards by swapping rows of the matrices
            xx_map[self.photon_hadamards, :], xz_map[self.photon_hadamards, :] = xz_map[self.photon_hadamards, :], xx_map[self.photon_hadamards, :]
            zx_map[self.photon_hadamards, :], zz_map[self.photon_hadamards, :] = zz_map[self.photon_hadamards, :], zx_map[self.photon_hadamards, :]

        return xx_map, xz_map, zx_map, zz_map

    def sample_qubit_errors_xyz(self, px, py, pz, n_samples, ts_samples=None):
        if ts_samples is None:
            ts_samples = self.sample_ts_errors_xyz(px, py, pz, num_ts=self.nq + 1, num_samples=n_samples)
#         qubit_xerr = (self.error_maps['xx'] @ ts_samples[0] + self.error_maps['zx'] @ ts_samples[1]) % 2
#         qubit_zerr = (self.error_maps['xz'] @ ts_samples[0] + self.error_maps['zz'] @ ts_samples[1]) % 2
        qubit_xerr = (np.dot(self.error_maps['xx'],ts_samples[0]) + np.dot(self.error_maps['zx'],ts_samples[1])) % 2
        qubit_zerr = (np.dot(self.error_maps['xz'],ts_samples[0]) + np.dot(self.error_maps['zz'],ts_samples[1])) % 2
        return qubit_xerr.astype(np.uint8), qubit_zerr.astype(np.uint8)



def resource_state_errors(rs_hadamards, num_rs, ex, ey, ez, n_samples, error_type='spin dephasing',
                          q_to_keep=None, hadamard_photons=True):
    """
    Shape this so that it is n x m x l array, where n = resource state indices, m = qubit index in resource state, l = sample index
    Importantly, return the thing that is flipped, not the error type! X error flips the Z outcome, so a X0 error would
    look like (0, 0, 0, ..., 1, 0, 0, ...)
    :param q_to_keep:
    :param rs_hadamards:
    :param num_rs:
    :param ex:
    :param ey:
    :param ez:
    :param n_samples:
    :param error_type: is defaulted to spin dephasing, no iid
    :return:
    """
    if error_type == 'spin dephasing':
        es = ErrorSampler(rs_hadamards, ex, ey, ez, good_q_ixs=q_to_keep, hadamard_photons=hadamard_photons)
        error_samples2d = es.sample_qubit_errors_xyz(ex, ey, ez, n_samples * num_rs)
    elif error_type == 'iid': # unused atm because there is no input for error_type
        num_qbts = len(rs_hadamards) + 1
        error_samples2d = np.random.choice((0, 1), p=(1-ex, ex), size=(num_qbts, n_samples * num_rs)), np.random.choice((0, 1), p=(1-ez, ez), size=(num_qbts, n_samples * num_rs))
    else:
        raise NotImplementedError
    if q_to_keep is not None:
        error_samples2d = error_samples2d[0][q_to_keep, :], error_samples2d[1][q_to_keep, :]
    # Reshape so that it is an nq x n_shots array. Remember indexes of qubits to resource states are the first n in rs1, the next n in rs2 etc.
    # Stack the X on top of the z, so it is in the usual form (X1, X2, X3, ..., Z1, Z2, Z3, ...)
    full_errors = [np.array([error_samples2d[ix][:, range(rs_ix * n_samples, (rs_ix+1) * n_samples)]
                                     for rs_ix in range(num_rs)]) for ix in (0, 1)] # added Sep 19 2024
    return full_errors

def encoded_chain_sampler(chain_length, ex, ey, ez, n_samples, num_chains, code_size=4, extra_chain_length=6, xx_eras_rate=0., zz_eras_rate=0.0):
    rs_hadamards = ([0] * (code_size-1) + [1]) * (chain_length + extra_chain_length) + [0]
    tot_num_q = code_size * (chain_length + extra_chain_length)
    good_ixs = [i for i in range(tot_num_q) if (code_size * extra_chain_length / 2) - 1 < i % tot_num_q < tot_num_q - (code_size * extra_chain_length / 2)]
    rs_flips = resource_state_errors(rs_hadamards, num_rs=num_chains, ex=ex, ey=ey, ez=ez, n_samples=n_samples, hadamard_photons=False, q_to_keep=good_ixs)
    return rs_flips

def branched_chains_sampler(chain_length, ex, ez, n_samples, num_chains):
    rs_hadamards =  [0, 1] * (chain_length + 5) + [0]
    rs_flips = resource_state_errors(rs_hadamards, num_rs=num_chains, ex=ex, ez=ez, n_samples=n_samples)
    return np.array(rs_flips)[:,:,3: chain_length + 3,:] # added 5 Mar 2024, dimension: (2, num_res_states, num_qubits_res_state, n_samples)


if __name__ == '__main__':
    flips = encoded_chain_sampler(100, 0.1, 0.1, 0.1, 2, 1, 4, 6)
    print(f'{len(flips)=}')
    print(np.sum(flips[:400]))
    print(np.sum(flips[400:]))

    error_sampler = ErrorSampler([1] * 11, good_q_ixs=list(range(1, 11)))
    mats = error_sampler.construct_ts_rs_map()
    print("\n")
    for m in mats:
        print(np.where(m))

    n_samples=100000
    chain_length = 30
    num_chains = 2
    errors_branched_chains = branched_chains_sampler(chain_length, 0.01, 0.01, n_samples, num_chains)
    print(errors_branched_chains)
    print(errors_branched_chains.shape)
    print(np.sum(errors_branched_chains, axis=1)[:2 * chain_length * num_chains]/n_samples)
    print(np.sum(errors_branched_chains, axis=1)[2 * chain_length * num_chains:]/n_samples)

    # define hadamard orders for some common emitter RSGs
    linear_6 = [1] * 5
    star4 = [0, 0, 0]
    linear_8 = [1] * 7
    branched6 = [1, 0, 1, 0, 1, 1]
    es = ErrorSampler(linear_8,  0.01, 0.01)

    # Takes 100k samples for 8q linear cluster in 0.1s
    t0 = time()
    q_samples = es.sample_qubit_errors(100000)
    t1 = time()
    print(t1-t0)

    # To generate 1000 samples for 1000 4 qubit star graph states takes <1s
    t0 = time()
    errors = resource_state_errors(star4, 1000, 0.01, 0.01, 1000)
    print(time() - t0)
