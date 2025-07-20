import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon



#################### sFFCC Lattice class ####################

class FFCCLattice_chain(object):
    def __init__(self, x_size, y_size=None, z_size=None):
        ## boundary is considered to be periodic for simplicity

        self.lattice_x_size = x_size
        if y_size is not None:
            self.lattice_y_size = y_size
        else:
            self.lattice_y_size = x_size
        if z_size is not None:
            self.lattice_z_size = z_size
        else:
            self.lattice_z_size = x_size

        self.num_cells = self.lattice_x_size * self.lattice_y_size * self.lattice_z_size
        self.data_type, self.empty_ix = self.get_data_type()

        # 3d spatial shifts to move between lattice cells. Defines the lattice cartesian system
        self.cell_xshift = np.array((1, 0, 0))
        self.cell_yshift = np.array((0.5, np.sqrt(3.)/2., 0))
        self.cell_zshift = np.array((0, 0, 1))

        # Shift to move between qubits in the lattice.
        # These correspond to the half length of exagon sides (in x and y) and to distance between layers of the code.
        self.xshift = self.cell_xshift / 3
        self.yshift = self.cell_yshift / 3
        self.zshift = self.cell_zshift / 6  # z shift between different layers of the code.

        self.cells_shapes = []
        self.syns_shapes = []

        self.fusions_positions = []
        self.qbts_positions = []

        self.num_fusions = 0
        self.tot_num_fusions = 0

        self.num_physical_qubits = 0

        self.fusion_biases = None

        self.fusions_timeorder = None

        self.current_fus_ix = 0
        
        ##############################
        self.num_fus_in_syndrome = 12 # syndrome is similar to a unit cell
        self.num_fus_in_cell = 18 # number of labels for fusions in a unit cell that does not repeat
        self.num_res_state_in_cell = 6 # number of labels for resource states in a unit cell.
        self.num_qubits_res_state = 6 # numer of photons/ qubits that undergo fusion in a resource state
        self.num_qubits_in_cell = self.num_res_state_in_cell*self.num_qubits_res_state # number of photons (being fused) in a unit cell
        self.num_fusions = self.num_fus_in_cell * self.num_cells # total number of fusions 
        self.num_physical_qubits = self.num_qubits_in_cell * self.num_cells # total number of physical qubits that undergo fusion
        self.num_res_states = self.num_res_state_in_cell * self.lattice_x_size * self.lattice_y_size # self.num_res_state_in_cell * self.num_cells # total number of resource states
        ############################

        # This is structured as:
        # [[fusions in cell 1], [fusions in cell 2], ... ]
        # dimension: no. of cells x no. of fusions per cell
        self.cells_fusions_struct = np.array([[self.empty_ix] * self.num_fus_in_cell]*self.num_cells, dtype=self.data_type)

        # This is structured as:
        # [[qubits in cell 1], [qubits in cell 2], ... ]
        self.cells_qbts_struct = np.array([[qbt_ix + self.num_qubits_in_cell*cell_ix for qbt_ix in range(self.num_qubits_in_cell)]
                                           for cell_ix in range(self.num_cells)], dtype=self.data_type)

        # This is structured as:
        # [[[fusions in red primal synd cell1], [fusions in green primal synd cell1], [fusions in blue primal synd cell1]],
        #  [[fusions in red primal synd cell2], [fusions in green primal synd cell2], [fusions in blue primal synd cell2]], ...]
        # The syndromes in each cell are arranged as:
        #                                                    /
        #                                               ____/  blue
        #                                              /    \
        #                                             /  red \_____
        #                                             \      /
        #                                              \____/  green
        #                                                   \
        #                                                    \
        self.cells_primal_syndrs_struct = np.array([[[self.empty_ix]*self.num_fus_in_syndrome]*3]*self.num_cells, dtype=self.data_type)
        self.cells_dual_syndrs_struct = np.array([[[self.empty_ix]*self.num_fus_in_syndrome]*3]*self.num_cells, dtype=self.data_type)


        ############################################################
        ##### Parts that relate the physical qubits to fusions
        ############################################################

        # self.qbts_in_fusions relates which physical qubits (photons) enter each fusion
        # This is structured as:
        # [[qubit1 in fusion 1, qubit2 in fusion 1], [qubit1 in fusion 2, qubit2 in fusion 2], ...]
        # Where qubit-i is the index of physical qubits in the lattice

        self.qbts_in_fusions = np.array([[self.empty_ix, self.empty_ix]]*(self.num_fus_in_cell*self.num_cells), dtype=self.data_type)


        # self.fusion_for_qubits says in which fusion each qubit enters (it is the inverse of self.qbts_in_fusions )
        # This is structured as:
        # [fusion index for qubit1, fusion index for qubit2, ...]
        # Where qubit-i is the index of physical qubits in the lattice

        self.fusion_for_qubits = np.array([self.empty_ix]*(self.num_qubits_in_cell*self.num_cells), dtype=self.data_type)

        # self.resource_states_in_cells relates the resource states that are in each layer of each cell
        # This is structured as:
        # [
        #  [
        #   [black index in layer0 of cell0, orange index in layer0 of cell0, purple index in layer0 of cell0],
        #   [black index in layer1 of cell0, orange index in layer1 of cell0, purple index in layer1 of cell0],
        #    ...
        #  ],
        #  [
        #   [black index in layer0 of cell1, orange index in layer0 of cell1, purple index in layer0 of cell1],
        #   [black index in layer1 of cell1, orange index in layer1 of cell1, purple index in layer1 of cell1],
        #    ...
        #  ],
        #  ...
        # [
        # # dimension: (no. of cells, no. of layers per cell, no. of colors/res states)
#         self.resource_states_in_cells = np.array([[[self.empty_ix]*3]*6]*self.num_cells, dtype=self.data_type)
#=================================================================================
        # dimension: (no. of cells, no. of res states per cell)
        self.resource_states_in_cells = np.array([[self.empty_ix]*self.num_res_state_in_cell]*self.num_cells, dtype=self.data_type)



        # self.qbts_in_resource_states relates the physical qubits (photons) labels
        # to the qubit indices in each resource state
        # This is structured as:
        # [
        #   [
        #     [qubit index for photon1 in black chain1, qubit index for photon2 in black chain1, ...],
        #     [qubit index for photon1 in orange chain1, qubit index for photon2 in orange chain1, ...]
        #     [qubit index for photon1 in purple chain1, qubit index for photon2 in purple chain1, ...]
        #   ],
        #   [
        #     [qubit index for photon1 in black chain2, qubit index for photon2 in black chain2, ...],
        #     [qubit index for photon1 in orange chain2, qubit index for photon2 in orange chain2, ...]
        #     [qubit index for photon1 in purple chain2, qubit index for photon2 in purple chain2, ...]
        #   ], ...
        #
        #]
        #
        # Where qubit-i is the index of physical qubits in the lattice
        
#=================================================================================
        # # dimension: (no. of cells in x,y, no. of res state, no. of qubits per res state)
#         self.qbts_in_resource_states = np.array([[[self.empty_ix]*(self.num_qubits_res_state*self.lattice_z_size)]*self.num_res_state_in_cell]
#                                                 * (self.lattice_x_size*self.lattice_y_size), dtype=self.data_type)
        # # dimension: (total no. of res state, no. of qubits per res state)
        self.qbts_in_resource_states = np.array([[self.empty_ix]*(self.num_qubits_res_state*self.lattice_z_size)]*self.num_res_state_in_cell
                                                * (self.lattice_x_size*self.lattice_y_size), dtype=self.data_type)
#=================================================================================
        # self.resource_state_of_qbt contains the resource state of each physical qbt (it is the inverse of self.qbts_in_resource_states).
        # This is structured as:
        # [ [color index of resource state of qbt1, chain index, qbt index in chain for qbt1]
        #   [color index of resource state of qbt1, chain index, qbt index in chain for qbt1], ... ]
        # Where the color indexe are: 0 -> black chains, 1 -> orange chains, 2 -> purple chains
        self.resource_state_of_qbt = np.array([[self.empty_ix, self.empty_ix]]*(self.num_qubits_res_state*self.num_res_state_in_cell*self.num_cells),
                                              dtype=self.data_type)
        
#         # List that says if each qubit is measured as a Z in the primal fusion outcome
#         self.qubits_with_z_in_primal = np.zeros(self.num_qubits_in_cell*self.num_cells, dtype = np.uint8) # assume XZ fusion
#         self.qubits_with_z_in_dual = np.zeros(self.num_qubits_in_cell*self.num_cells, dtype = np.uint8) # assume ZX fusion
        #############################################################


        self.log_ops_fusions = [None, None, None]  # logical ops for x-like, y-like, and z-like surfaces
        self.log_ops_fusions_dual = [None, None, None]  # logical ops for x-like, y-like, and z-like surfaces, dual

        self.build_lattice_structures()
        self.build_resource_states()
        self.associate_qubits_to_resources()
        # self.build_lattice_edges()
        self.build_primal_syndromes()
        self.build_dual_syndromes()
        self.get_logical_operators()
        self.get_fusions_primal_isZZ()
        self.get_fusions_layer_order()

    def build_lattice_structures(self):
        # Starts building the lattice
        for z_ix in range(self.lattice_z_size):
            for y_ix in range(self.lattice_y_size):
                for x_ix in range(self.lattice_x_size):
                    # cell_pos = (x_ix + 0.5) * self.cell_xshift + (y_ix + 0.5) * self.cell_yshift + \
                    #            (z_ix + 0.5) * self.cell_zshift
                    cell_ix = self.cell_xyzcoords_to_ix([x_ix, y_ix, z_ix])

                    # Add primal fusions
                    # Layer 0Red
                    self.cells_fusions_struct[cell_ix][0] = self.current_fus_ix
                    neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
                    neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
                    self.qbts_in_fusions[self.current_fus_ix][0] = 0 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 3 + self.num_qubits_in_cell*neigh_cell
                    self.current_fus_ix += 1

                    self.cells_fusions_struct[cell_ix][1] = self.current_fus_ix
                    neigh_cell = self.shifted_cell_ix(cell_ix, -1, 'y')
                    self.qbts_in_fusions[self.current_fus_ix][0] = 1 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 4 + self.num_qubits_in_cell*neigh_cell
                    self.current_fus_ix += 1

                    self.cells_fusions_struct[cell_ix][2] = self.current_fus_ix
                    neigh_cell = self.shifted_cell_ix(cell_ix, -1, 'x')
                    self.qbts_in_fusions[self.current_fus_ix][0] = 2 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 5 + self.num_qubits_in_cell*neigh_cell
                    self.current_fus_ix += 1

                    # Layer 1Green
                    self.cells_fusions_struct[cell_ix][3] = self.current_fus_ix
                    self.qbts_in_fusions[self.current_fus_ix][0] = 6 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 7 + self.num_qubits_in_cell*cell_ix
                    self.current_fus_ix += 1

                    self.cells_fusions_struct[cell_ix][4] = self.current_fus_ix
                    self.qbts_in_fusions[self.current_fus_ix][0] = 8 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 9 + self.num_qubits_in_cell*cell_ix
                    self.current_fus_ix += 1

                    self.cells_fusions_struct[cell_ix][5] = self.current_fus_ix
                    self.qbts_in_fusions[self.current_fus_ix][0] = 10 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 11 + self.num_qubits_in_cell*cell_ix
                    self.current_fus_ix += 1

                    # Layer 2Blue
                    self.cells_fusions_struct[cell_ix][6] = self.current_fus_ix
                    self.qbts_in_fusions[self.current_fus_ix][0] = 12 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 17 + self.num_qubits_in_cell*cell_ix
                    self.current_fus_ix += 1

                    self.cells_fusions_struct[cell_ix][7] = self.current_fus_ix
                    self.qbts_in_fusions[self.current_fus_ix][0] = 13 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 14 + self.num_qubits_in_cell*cell_ix
                    self.current_fus_ix += 1

                    self.cells_fusions_struct[cell_ix][8] = self.current_fus_ix
                    self.qbts_in_fusions[self.current_fus_ix][0] = 15 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 16 + self.num_qubits_in_cell*cell_ix
                    self.current_fus_ix += 1

                    # Layer 3Red
                    self.cells_fusions_struct[cell_ix][9] = self.current_fus_ix
                    neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
                    neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
                    self.qbts_in_fusions[self.current_fus_ix][0] = 18 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 21 + self.num_qubits_in_cell*neigh_cell
                    self.current_fus_ix += 1

                    self.cells_fusions_struct[cell_ix][10] = self.current_fus_ix
                    neigh_cell = self.shifted_cell_ix(cell_ix, -1, 'y')
                    self.qbts_in_fusions[self.current_fus_ix][0] = 19 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 22 + self.num_qubits_in_cell*neigh_cell
                    self.current_fus_ix += 1

                    self.cells_fusions_struct[cell_ix][11] = self.current_fus_ix
                    neigh_cell = self.shifted_cell_ix(cell_ix, -1, 'x')
                    self.qbts_in_fusions[self.current_fus_ix][0] = 20 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 23 + self.num_qubits_in_cell*neigh_cell
                    self.current_fus_ix += 1

                    # Layer 4Green
                    self.cells_fusions_struct[cell_ix][12] = self.current_fus_ix
                    self.qbts_in_fusions[self.current_fus_ix][0] = 24 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 25 + self.num_qubits_in_cell*cell_ix
                    self.current_fus_ix += 1

                    self.cells_fusions_struct[cell_ix][13] = self.current_fus_ix
                    self.qbts_in_fusions[self.current_fus_ix][0] = 26 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 27 + self.num_qubits_in_cell*cell_ix
                    self.current_fus_ix += 1

                    self.cells_fusions_struct[cell_ix][14] = self.current_fus_ix
                    self.qbts_in_fusions[self.current_fus_ix][0] = 28 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 29 + self.num_qubits_in_cell*cell_ix
                    self.current_fus_ix += 1

                    # Layer 5Blue
                    self.cells_fusions_struct[cell_ix][15] = self.current_fus_ix
                    self.qbts_in_fusions[self.current_fus_ix][0] = 30 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 35 + self.num_qubits_in_cell*cell_ix
                    self.current_fus_ix += 1

                    self.cells_fusions_struct[cell_ix][16] = self.current_fus_ix
                    self.qbts_in_fusions[self.current_fus_ix][0] = 31 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 32 + self.num_qubits_in_cell*cell_ix
                    self.current_fus_ix += 1

                    self.cells_fusions_struct[cell_ix][17] = self.current_fus_ix
                    self.qbts_in_fusions[self.current_fus_ix][0] = 33 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_fusions[self.current_fus_ix][1] = 34 + self.num_qubits_in_cell*cell_ix
                    self.current_fus_ix += 1

        for fus_ix, fus_qbts in enumerate(self.qbts_in_fusions):
            self.fusion_for_qubits[fus_qbts[0]] = fus_ix
            self.fusion_for_qubits[fus_qbts[1]] = fus_ix



    def build_resource_states(self):
        for z_ix in range(self.lattice_z_size):
            for y_ix in range(self.lattice_y_size):
                for x_ix in range(self.lattice_x_size):
                    cell_ix = self.cell_xyzcoords_to_ix([x_ix, y_ix, z_ix])
                    # self.resource_states_in_cells dimension: (n_cell, no. of res state in cell)
                 
                    # dimension: (no. of cells, no. of res states per cell)
                    self.resource_states_in_cells[cell_ix][0] = self.num_res_state_in_cell*self.cell_xyzcoords_to_ix([x_ix, y_ix, 0]) + 0 # in z, they overlap
                    self.resource_states_in_cells[cell_ix][1] = self.num_res_state_in_cell*self.cell_xyzcoords_to_ix([x_ix, y_ix, 0]) + 1 # in z, they overlap
                    self.resource_states_in_cells[cell_ix][2] = self.num_res_state_in_cell*self.cell_xyzcoords_to_ix([x_ix, y_ix, 0]) + 2 # in z, they overlap
                    self.resource_states_in_cells[cell_ix][3] = self.num_res_state_in_cell*self.cell_xyzcoords_to_ix([x_ix, y_ix, 0]) + 3 # in z, they overlap
                    self.resource_states_in_cells[cell_ix][4] = self.num_res_state_in_cell*self.cell_xyzcoords_to_ix([x_ix, y_ix, 0]) + 4 # in z, they overlap
                    self.resource_states_in_cells[cell_ix][5] = self.num_res_state_in_cell*self.cell_xyzcoords_to_ix([x_ix, y_ix, 0]) + 5 # in z, they overlap

    def associate_qubits_to_resources(self):
        for z_ix in range(self.lattice_z_size):
            for y_ix in range(self.lattice_y_size):
                for x_ix in range(self.lattice_x_size):
                    cell_ix = self.cell_xyzcoords_to_ix([x_ix, y_ix, z_ix])
                     
                     # # dimension: (no. of cells in x,y, no. of res state in cell, no. of qubits per res state)
#         self.qbts_in_resource_states = np.array([[[self.empty_ix]*(self.num_qubits_res_state*self.lattice_z_size)]*self.num_res_state_in_cell]
#                                                 * (self.lattice_x_size*self.lattice_y_size), dtype=self.data_type)

                    # in total 6*Lz qubits in the resource state.
                    # assign qubit index in cell to each qubit in the resource state
                    # Starting from layer 0, cell (x,y) = (0,0)
                    
                    #######################################################
                    # cells_qbts_struct[cell_ix][]: [[qubits in cell 1], [qubits in cell 2], ... ]

                    self.cells_qbts_struct[cell_ix] = np.array([i + self.num_qubits_in_cell*cell_ix for i in range(self.num_qubits_in_cell)], dtype = self.data_type)  # dimension: no. of cells x number of qubits in cell
#                     self.qubits_with_z_in_primal[self.cells_qbts_struct[cell_ix][[6,7,8,9,10,17,18,19,20,21,22,29]+self.num_qubits_res_state*(z_ix-1)*[1] * 12]] = 1 # 6,7,8,9,...,29 qbts are measured in Z for primal fusion syndrome
#                     self.qubits_with_z_in_dual[self.cells_qbts_struct[cell_ix][[12,13,14,15,16,23,24,25,26,27,28,35]+self.num_qubits_res_state*(z_ix-1)*[1] * 12]] = 1 # 6,7,8,9,...,29 qbts are measured in Z for dual fusion syndrome
                    #######################################################
                    chain_ix = self.resource_states_in_cells[cell_ix][0] # extract res state index
                    self.qbts_in_resource_states[chain_ix][0 + self.num_qubits_res_state*z_ix] = 0 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][1 + self.num_qubits_res_state*z_ix] = 6 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][2 + self.num_qubits_res_state*z_ix] = 12 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][3 + self.num_qubits_res_state*z_ix] = 18 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][4 + self.num_qubits_res_state*z_ix] = 24 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][5 + self.num_qubits_res_state*z_ix] = 30 + self.num_qubits_in_cell*cell_ix

                    chain_ix = self.resource_states_in_cells[cell_ix][1] # extract res state index
                    self.qbts_in_resource_states[chain_ix][0 + self.num_qubits_res_state*z_ix] = 1 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][1 + self.num_qubits_res_state*z_ix] = 7 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][2 + self.num_qubits_res_state*z_ix] = 13 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][3 + self.num_qubits_res_state*z_ix] = 19 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][4 + self.num_qubits_res_state*z_ix] = 25 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][5 + self.num_qubits_res_state*z_ix] = 31 + self.num_qubits_in_cell*cell_ix

                    chain_ix = self.resource_states_in_cells[cell_ix][2] # extract res state index
                    self.qbts_in_resource_states[chain_ix][0 + self.num_qubits_res_state*z_ix] = 2 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][1 + self.num_qubits_res_state*z_ix] = 8 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][2 + self.num_qubits_res_state*z_ix] = 14 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][3 + self.num_qubits_res_state*z_ix] = 20 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][4 + self.num_qubits_res_state*z_ix] = 26 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][5 + self.num_qubits_res_state*z_ix] = 32 + self.num_qubits_in_cell*cell_ix
                    
                    chain_ix = self.resource_states_in_cells[cell_ix][3] # extract res state index
                    self.qbts_in_resource_states[chain_ix][0 + self.num_qubits_res_state*z_ix] = 3 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][1 + self.num_qubits_res_state*z_ix] = 9 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][2 + self.num_qubits_res_state*z_ix] = 15 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][3 + self.num_qubits_res_state*z_ix] = 21 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][4 + self.num_qubits_res_state*z_ix] = 27 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][5 + self.num_qubits_res_state*z_ix] = 33 + self.num_qubits_in_cell*cell_ix
            
                    chain_ix = self.resource_states_in_cells[cell_ix][4] # extract res state index
                    self.qbts_in_resource_states[chain_ix][0 + self.num_qubits_res_state*z_ix] = 4 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][1 + self.num_qubits_res_state*z_ix] = 10 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][2 + self.num_qubits_res_state*z_ix] = 16 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][3 + self.num_qubits_res_state*z_ix] = 22 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][4 + self.num_qubits_res_state*z_ix] = 28 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][5 + self.num_qubits_res_state*z_ix] = 34 + self.num_qubits_in_cell*cell_ix

                    chain_ix = self.resource_states_in_cells[cell_ix][5] # extract res state index
                    self.qbts_in_resource_states[chain_ix][0 + self.num_qubits_res_state*z_ix] = 5 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][1 + self.num_qubits_res_state*z_ix] = 11 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][2 + self.num_qubits_res_state*z_ix] = 17 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][3 + self.num_qubits_res_state*z_ix] = 23 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][4 + self.num_qubits_res_state*z_ix] = 29 + self.num_qubits_in_cell*cell_ix
                    self.qbts_in_resource_states[chain_ix][5 + self.num_qubits_res_state*z_ix] = 35 + self.num_qubits_in_cell*cell_ix

        for chain_ix in range(self.qbts_in_resource_states.shape[0]):
            for qbt_ix in range(self.qbts_in_resource_states.shape[1]):
                self.resource_state_of_qbt[self.qbts_in_resource_states[chain_ix][qbt_ix]] = \
                    np.array([chain_ix, qbt_ix], dtype=self.data_type)              
#         for color_ix in range(self.qbts_in_resource_states.shape[1]):
#             for qbt_ix in range(self.qbts_in_resource_states.shape[2]):
#                 self.resource_state_of_qbt[self.qbts_in_resource_states[chain_ix][color_ix][qbt_ix]] = \
#                     np.array([color_ix, chain_ix, qbt_ix], dtype=self.data_type)


    def build_primal_syndromes(self):

        for cell_ix, fusions in enumerate(self.cells_fusions_struct):
            # Red syndrome
            self.cells_primal_syndrs_struct[cell_ix][0][0] = fusions[3]
            self.cells_primal_syndrs_struct[cell_ix][0][1] = fusions[4]
            self.cells_primal_syndrs_struct[cell_ix][0][2] = fusions[5]
            self.cells_primal_syndrs_struct[cell_ix][0][3] = fusions[6]
            self.cells_primal_syndrs_struct[cell_ix][0][4] = fusions[7]
            self.cells_primal_syndrs_struct[cell_ix][0][5] = fusions[8]
            self.cells_primal_syndrs_struct[cell_ix][0][6] = fusions[12]
            self.cells_primal_syndrs_struct[cell_ix][0][7] = fusions[13]
            self.cells_primal_syndrs_struct[cell_ix][0][8] = fusions[14]
            self.cells_primal_syndrs_struct[cell_ix][0][9] = fusions[15]
            self.cells_primal_syndrs_struct[cell_ix][0][10] = fusions[16]
            self.cells_primal_syndrs_struct[cell_ix][0][11] = fusions[17]

            # Green syndrome
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][1][0] = self.cells_fusions_struct[neigh_cell][17]

            neigh_cell = self.shifted_cell_ix(cell_ix, -1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][1][1] = self.cells_fusions_struct[neigh_cell][15]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][1][2] = self.cells_fusions_struct[neigh_cell][16]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][1][3] = self.cells_fusions_struct[neigh_cell][1]

            self.cells_primal_syndrs_struct[cell_ix][1][4] = fusions[0]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][1][5] = self.cells_fusions_struct[neigh_cell][2]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][1][6] = self.cells_fusions_struct[neigh_cell][8]

            self.cells_primal_syndrs_struct[cell_ix][1][7] = fusions[6]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][1][8] = self.cells_fusions_struct[neigh_cell][7]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][1][9] = self.cells_fusions_struct[neigh_cell][10]

            self.cells_primal_syndrs_struct[cell_ix][1][10] = fusions[9]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][1][11] = self.cells_fusions_struct[neigh_cell][11]

            # Blue syndrome
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][2][0] = self.cells_fusions_struct[neigh_cell][11]
            
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][2][1] = self.cells_fusions_struct[neigh_cell][10]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][2][2] = self.cells_fusions_struct[neigh_cell][9]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_primal_syndrs_struct[cell_ix][2][3] = self.cells_fusions_struct[neigh_cell][13]

            self.cells_primal_syndrs_struct[cell_ix][2][4] = fusions[14]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            self.cells_primal_syndrs_struct[cell_ix][2][5] = self.cells_fusions_struct[neigh_cell][12]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][2][6] = self.cells_fusions_struct[neigh_cell][2]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][2][7] = self.cells_fusions_struct[neigh_cell][1]
            
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][2][8] = self.cells_fusions_struct[neigh_cell][0]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][2][9] = self.cells_fusions_struct[neigh_cell][4]
            
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][2][10] = self.cells_fusions_struct[neigh_cell][5]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'z')
            self.cells_primal_syndrs_struct[cell_ix][2][11] = self.cells_fusions_struct[neigh_cell][3]

    def build_dual_syndromes(self):

        for cell_ix, fusions in enumerate(self.cells_fusions_struct):
            # Red syndrome
            neigh_cell = self.shifted_cell_ix(cell_ix, -1, 'z')
            self.cells_dual_syndrs_struct[cell_ix][0][0] = self.cells_fusions_struct[neigh_cell][12]
            self.cells_dual_syndrs_struct[cell_ix][0][1] = self.cells_fusions_struct[neigh_cell][13]
            self.cells_dual_syndrs_struct[cell_ix][0][2] = self.cells_fusions_struct[neigh_cell][14]
            self.cells_dual_syndrs_struct[cell_ix][0][3] = self.cells_fusions_struct[neigh_cell][15]
            self.cells_dual_syndrs_struct[cell_ix][0][4] = self.cells_fusions_struct[neigh_cell][16]
            self.cells_dual_syndrs_struct[cell_ix][0][5] = self.cells_fusions_struct[neigh_cell][17]
            
            self.cells_dual_syndrs_struct[cell_ix][0][6] = fusions[3]
            self.cells_dual_syndrs_struct[cell_ix][0][7] = fusions[4]
            self.cells_dual_syndrs_struct[cell_ix][0][8] = fusions[5]
            self.cells_dual_syndrs_struct[cell_ix][0][9] = fusions[6]
            self.cells_dual_syndrs_struct[cell_ix][0][10] = fusions[7]
            self.cells_dual_syndrs_struct[cell_ix][0][11] = fusions[8]

            # Green syndrome
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
            self.cells_dual_syndrs_struct[cell_ix][1][0] = self.cells_fusions_struct[neigh_cell][8]
            
            self.cells_dual_syndrs_struct[cell_ix][1][1] = fusions[6]
            
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_dual_syndrs_struct[cell_ix][1][2] = self.cells_fusions_struct[neigh_cell][7]
            
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_dual_syndrs_struct[cell_ix][1][3] = self.cells_fusions_struct[neigh_cell][10]
            
            self.cells_dual_syndrs_struct[cell_ix][1][4] = fusions[9]
            
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_dual_syndrs_struct[cell_ix][1][5] = self.cells_fusions_struct[neigh_cell][11]
            
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, -1, 'y')
            self.cells_dual_syndrs_struct[cell_ix][1][6] = self.cells_fusions_struct[neigh_cell][17]
            
            self.cells_dual_syndrs_struct[cell_ix][1][7] = fusions[15]
            
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_dual_syndrs_struct[cell_ix][1][8] = self.cells_fusions_struct[neigh_cell][16]
            
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'z')
            self.cells_dual_syndrs_struct[cell_ix][1][9] = self.cells_fusions_struct[neigh_cell][1]
            
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'z')
            self.cells_dual_syndrs_struct[cell_ix][1][10] = self.cells_fusions_struct[neigh_cell][0]
            
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            neigh_cell = self.shifted_cell_ix(neigh_cell, +1, 'z')
            self.cells_dual_syndrs_struct[cell_ix][1][11] = self.cells_fusions_struct[neigh_cell][2]

            # Blue syndrome
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_dual_syndrs_struct[cell_ix][2][0] = self.cells_fusions_struct[neigh_cell][2]
            
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            self.cells_dual_syndrs_struct[cell_ix][2][1] = self.cells_fusions_struct[neigh_cell][1]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            self.cells_dual_syndrs_struct[cell_ix][2][2] = self.cells_fusions_struct[neigh_cell][0]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_dual_syndrs_struct[cell_ix][2][3] = self.cells_fusions_struct[neigh_cell][4]

            self.cells_dual_syndrs_struct[cell_ix][2][4] = fusions[5]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            self.cells_dual_syndrs_struct[cell_ix][2][5] = self.cells_fusions_struct[neigh_cell][3]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_dual_syndrs_struct[cell_ix][2][6] = self.cells_fusions_struct[neigh_cell][11]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            self.cells_dual_syndrs_struct[cell_ix][2][7] = self.cells_fusions_struct[neigh_cell][10]
            
            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            self.cells_dual_syndrs_struct[cell_ix][2][8] = self.cells_fusions_struct[neigh_cell][9]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'x')
            self.cells_dual_syndrs_struct[cell_ix][2][9] = self.cells_fusions_struct[neigh_cell][13]
            
            self.cells_dual_syndrs_struct[cell_ix][2][10] = fusions[14]

            neigh_cell = self.shifted_cell_ix(cell_ix, +1, 'y')
            self.cells_dual_syndrs_struct[cell_ix][2][11] = self.cells_fusions_struct[neigh_cell][12]



        # Function to calculate the logical operators of the lattice

    def get_logical_operators(self):

        # Primal

        log_x = []
        log_y = []
        log_z = []

        # build logical operator surface on z direction
        z_ix = 0
        for y_ix in range(self.lattice_y_size):
            for x_ix in range(self.lattice_x_size):
                cell_ix = self.cell_xyzcoords_to_ix((x_ix, y_ix, z_ix))
                fusions = self.cells_fusions_struct[cell_ix]
                log_z += list(fusions[[3, 4, 5, 6, 7, 8, 9, 10, 11]])

        
        # build logical operator surface on x direction (NOTE ONLY WORKS FOR EVEN L)
        x_ix = 0
        y_ix = 0
        step_ix = 0
        while x_ix != 0 or (y_ix != 0) or step_ix == 0:
            if step_ix % 2 == 0:
                for z_ix in range(self.lattice_z_size):
                    cell_ix = self.cell_xyzcoords_to_ix((x_ix, y_ix, z_ix))
                    fusions = self.cells_fusions_struct[cell_ix]
                    log_x += list(fusions[[0, 2, 3, 7, 9, 11, 12, 16]])
                x_ix = (x_ix - 1) % self.lattice_x_size
            else:

                for z_ix in range(self.lattice_z_size):
                    cell_ix = self.cell_xyzcoords_to_ix((x_ix, y_ix, z_ix))
                    fusions = self.cells_fusions_struct[cell_ix]
                    log_x += list(fusions[[5, 8, 14, 17]])
                y_ix = (y_ix + 1) % self.lattice_y_size
                x_ix = (x_ix - 1) % self.lattice_x_size

            step_ix += 1
            
        # build logical operator surface on y direction
        x_ix = 0
        y_ix = 0
        step_ix = 0 # cell number. odd cell, even cell
        while x_ix != 0 or (y_ix != 0) or step_ix == 0: # terminates when x_ix = y_ix = 0 goes back to starting point
            if step_ix % 2 == 0:
                for z_ix in range(self.lattice_z_size):
                    cell_ix = self.cell_xyzcoords_to_ix((x_ix, y_ix, z_ix))
                    fusions = self.cells_fusions_struct[cell_ix]
                    log_y += list(fusions[[0, 6, 14, 9, 15, 5]])

                y_ix = (y_ix + 1) % self.lattice_y_size

            else:
                for z_ix in range(self.lattice_z_size):
                    cell_ix = self.cell_xyzcoords_to_ix((x_ix, y_ix, z_ix))
                    fusions = self.cells_fusions_struct[cell_ix]
                    log_y += list(fusions[[1, 7, 13, 10, 16, 4]])

                x_ix = (x_ix - 1) % self.lattice_x_size
                y_ix = (y_ix + 1) % self.lattice_y_size
            step_ix += 1   
            
        self.log_ops_fusions = [log_x, log_y, log_z]

        # Dual

        log_x = []
        log_y = []
        log_z = []

        # build logical operator surface on z direction
        z_ix = 0
        for y_ix in range(self.lattice_y_size):
            for x_ix in range(self.lattice_x_size):
                cell_ix = self.cell_xyzcoords_to_ix((x_ix, y_ix, z_ix))
                fusions = self.cells_fusions_struct[cell_ix]
                log_z += list(fusions[[0, 1, 2, 3, 4, 5, 6, 7, 8]])

        
        # build logical operator surface on x direction
        x_ix = 0
        y_ix = 0
        step_ix = 0
        while x_ix != 0 or (y_ix != 0) or step_ix == 0:
            if step_ix % 2 == 0:
                for z_ix in range(self.lattice_z_size):
                    cell_ix = self.cell_xyzcoords_to_ix((x_ix, y_ix, z_ix))
                    fusions = self.cells_fusions_struct[cell_ix]
                    log_x += list(fusions[[0, 2, 3, 7, 9, 11, 12, 16]])
                x_ix = (x_ix - 1) % self.lattice_x_size
            else:

                for z_ix in range(self.lattice_z_size):
                    cell_ix = self.cell_xyzcoords_to_ix((x_ix, y_ix, z_ix))
                    fusions = self.cells_fusions_struct[cell_ix]
                    log_x += list(fusions[[5, 8, 14, 17]])
                y_ix = (y_ix + 1) % self.lattice_y_size
                x_ix = (x_ix - 1) % self.lattice_x_size

            step_ix += 1

        # build logical operator surface on y direction
        x_ix = 0
        y_ix = 0
        step_ix = 0 # cell number. odd cell, even cell
        while x_ix != 0 or (y_ix != 0) or step_ix == 0: # terminates when x_ix = y_ix = 0 goes back to starting point
            if step_ix % 2 == 0:
                for z_ix in range(self.lattice_z_size):
                    cell_ix = self.cell_xyzcoords_to_ix((x_ix, y_ix, z_ix))
                    fusions = self.cells_fusions_struct[cell_ix]
                    log_y += list(fusions[[0, 6, 14, 9, 15, 5]])

                y_ix = (y_ix + 1) % self.lattice_y_size

            else:
                for z_ix in range(self.lattice_z_size):
                    cell_ix = self.cell_xyzcoords_to_ix((x_ix, y_ix, z_ix))
                    fusions = self.cells_fusions_struct[cell_ix]
                    log_y += list(fusions[[1, 7, 13, 10, 16, 4]])

                x_ix = (x_ix - 1) % self.lattice_x_size
                y_ix = (y_ix + 1) % self.lattice_y_size
            step_ix += 1   
            
        self.log_ops_fusions_dual = [log_x, log_y, log_z]



    # Functions that build the matching matrix for syndromes of the lattice

    def get_matching_matrix(self, lattice_type = 'Primal'):
        h_matrix = np.zeros((3 * self.num_cells, self.num_fusions), dtype=np.uint8)
        if lattice_type == 'Primal':
            for cell_ix, cell_syndrs in enumerate(self.cells_primal_syndrs_struct):
                for syndr_ix, syndr_nodes_list in enumerate(cell_syndrs):
                    syndr_fusions = syndr_nodes_list[syndr_nodes_list != self.empty_ix]
                    h_matrix[3 * cell_ix + syndr_ix, syndr_fusions] = np.ones(len(syndr_fusions), dtype=np.uint8)
        elif lattice_type == 'Dual':
            for cell_ix, cell_syndrs in enumerate(self.cells_dual_syndrs_struct):
                for syndr_ix, syndr_nodes_list in enumerate(cell_syndrs):
                    syndr_fusions = syndr_nodes_list[syndr_nodes_list != self.empty_ix]
                    h_matrix[3 * cell_ix + syndr_ix, syndr_fusions] = np.ones(len(syndr_fusions), dtype=np.uint8)
        else:
            raise ValueError('lattice_type can only be in [\'Primal\', \'Dual\']')
        return h_matrix

#     # Function to build the fusion biases for different biasing configurations
#     def get_fusions_biases(self, bias_type='Primal'):
#         ### 0: bias on primal (failure happens only on dual outcomes), 1: bias on dual.
#         if bias_type == 'Primal':
#             self.fusion_biases = np.zeros(self.num_fusions, dtype=np.uint8)
#         elif bias_type == 'Dual':
#             self.fusion_biases = np.ones(self.num_fusions, dtype=np.uint8)
#         elif bias_type == 'Random':
#             self.fusion_biases = np.random.binomial(1, 0.5, self.num_fusions).astype(np.uint8)
#         elif bias_type == 'PassiveBias' or bias_type == 'ActiveSimple':
#             self.fusion_biases = np.zeros(self.num_fusions, dtype=np.uint8)
#             for fusions in self.cells_fusions_struct:
#                 self.fusion_biases[fusions[9:]] = 1
#         else:
#             raise ValueError(
#                 'bias_type can only be in [\'Primal\',\'Dual\',\'Random\',\'PassiveBias\',\'ActiveSimple]\'')

    # Function to build the fusion biases for different biasing configurations
    def get_fusions_primal_isZZ(self):
        ### 1 means fusion type is ZZ; 0 means XX
        self.fusions_primal_isZZ = np.zeros(self.num_fusions, dtype=np.uint8)
        for fusions in self.cells_fusions_struct:
            self.fusions_primal_isZZ[fusions[[3,4,5,9,10,11,15,16,17]]] = 1
    
    # Function for doing fusions layer by layer. Give each fusion a layer number
    def get_fusions_layer_order(self):
        self.fusions_layer_order = np.zeros(self.num_fusions, dtype=np.uint8)
        for i_layer in range(self.lattice_z_size): # for each z
            for i in range(self.num_qubits_res_state): # generates fusion indices for x-y fusions on the same layer
                fusion_ix_layer = [] # indices of all fusions belonging to the same layer
                for cell_ix in range(self.lattice_x_size*self.lattice_y_size):
                    cell_ix_ = cell_ix + i*self.lattice_x_size*self.lattice_y_size
                    fusion_ix_layer += list(self.cells_fusions_struct[cell_ix][0:3] + 3*i*np.ones(len(self.cells_fusions_struct[cell_ix][0:3]), dtype=np.uint8))
                fusion_ix_layer_ = np.array(fusion_ix_layer)+(i_layer*self.num_fus_in_cell*self.lattice_x_size*self.lattice_y_size)*np.ones(len(fusion_ix_layer), dtype=np.uint8)
                self.fusions_layer_order[list(fusion_ix_layer_)] = i+i_layer*self.num_qubits_res_state # set layer number



    # Functions to deal and move in between with cells in the lattice

    def cell_xyzcoords_to_ix(self, cell_xytcoords):
        return cell_xytcoords[0] + cell_xytcoords[1] * self.lattice_x_size + \
               cell_xytcoords[2] * self.lattice_x_size * self.lattice_y_size

    def cell_ix_to_xyzcoords(self, cell_ix):
        t_coord = int(cell_ix / (self.lattice_x_size * self.lattice_y_size))
        y_coord = int(cell_ix / self.lattice_x_size) % self.lattice_y_size
        x_coord = cell_ix % self.lattice_x_size
        return np.array((x_coord, y_coord, t_coord))

    def shifted_cell_ix(self, cell_ix, shift, shift_axis):
        #     Function to get the index of a cell obtained shifting an initial cell with 'cell_ix'
        #     by a integer (positive or negative) step 'shift' along an axis 'shift_axis' in x, y, or t.
        if not isinstance(shift, int):
            raise ValueError('The parameter shift can only be an integer (positive or negative)')

        if shift_axis == 'x':
            axis_label = 0
            size_lim = self.lattice_x_size
        elif shift_axis == 'y':
            axis_label = 1
            size_lim = self.lattice_y_size
        elif shift_axis == 'z':
            axis_label = 2
            size_lim = self.lattice_z_size
        else:
            raise ValueError('Shift axis can only be one of (x, y, or z)')
        temp_coords = self.cell_ix_to_xyzcoords(cell_ix)
        temp_coords[axis_label] = (temp_coords[axis_label] + shift) % size_lim
        return self.cell_xyzcoords_to_ix(temp_coords)

        # Data type handler

    def get_data_type(self):
        # est_num_fusions_log = np.log2(36 * self.num_cells)
        # if est_num_fusions_log < 8:
        #     data_type = np.uint8
        # elif est_num_fusions_log < 16:
        #     data_type = np.uint16
        # elif est_num_fusions_log < 32:
        #     data_type = np.uint32
        # else:
        #     data_type = np.uint64

        data_type = np.uint32 # Always using np.uint32 for compatibility with C++ noise sampling functions
        return data_type, np.iinfo(data_type).max


    ###################################
    ########## Drawings Functions
    ###################################

    def calculate_qubits_positions(self):
        self.qbts_positions = np.zeros((self.num_physical_qubits, 3))

        for z_ix in range(self.lattice_z_size):
            for y_ix in range(self.lattice_y_size):
                for x_ix in range(self.lattice_x_size):
                    cell_ix = self.cell_xyzcoords_to_ix([x_ix, y_ix, z_ix])
                    cell_shift_vector = x_ix*self.cell_xshift + y_ix*self.cell_yshift + z_ix*self.cell_zshift

                    for layer_ix in range(6):
                        self.qbts_positions[0+6*layer_ix+36*cell_ix] = + self.xshift - self.yshift + self.zshift*layer_ix + cell_shift_vector
                        self.qbts_positions[1+6*layer_ix+36*cell_ix] = - self.yshift + self.zshift*layer_ix + cell_shift_vector
                        self.qbts_positions[2+6*layer_ix+36*cell_ix] = - self.xshift + self.zshift*layer_ix + cell_shift_vector
                        self.qbts_positions[3+6*layer_ix+36*cell_ix] = - self.xshift + self.yshift + self.zshift*layer_ix + cell_shift_vector
                        self.qbts_positions[4+6*layer_ix+36*cell_ix] = + self.yshift + self.zshift*layer_ix + cell_shift_vector
                        self.qbts_positions[5+6*layer_ix+36*cell_ix] = + self.xshift + self.zshift*layer_ix + cell_shift_vector

    def draw_2d_colorcode(self, axis=None, fig_size=None):

        scale = self.xshift[0]

        Lx = self.lattice_x_size
        Ly = self.lattice_y_size

        edge_color = 'black'
        edge_alpha = 1
        edge_width = 1

        cell_color_red = 'red'
        cell_color_green = 'green'
        cell_color_blue = 'blue'
        cell_alpha = 0.2

        add_cell_labels = True
        cell_labels_color = 'red'
        cell_labels_alpha = 0.5

        r = scale
        a = r * np.sqrt(3) / 2.
        b = r / 2.

        x_len = 3 * (r * (Lx - 1) + b * (Ly - 1)) + 3 * r
        y_len = 3 * a * (Ly - 1) + 4 * a

        if axis == None:

            if fig_size == None:
                fig_size = 12


            fig_aspect = y_len / x_len

            fig = plt.figure(figsize=(fig_size, fig_aspect * fig_size))
            ax = fig.add_subplot()
        else:
            ax = axis

        cell_labels_size = 280 * scale / x_len

        tile_edges0 = np.array([[[-r, 0], [-b, a]],
                                [[-b, a], [b, a]],
                                [[b, a], [r, 0]],
                                [[r, 0], [b, -a]],
                                [[b, -a], [-b, -a]],
                                [[-b, -a], [-r, 0]],
                                [[b, a], [2 * b, 2 * a]],
                                [[r, 0], [2 * r, 0]],
                                [[b, -a], [2 * b, -2 * a]],
                                ])

        red_cell_points0 = np.array([[-r, 0], [-b, a], [b, a], [r, 0], [b, -a], [-b, -a]])
        green_cell_points0 = red_cell_points0 + np.array([[r + b, -a]] * 6)
        blue_cell_points0 = red_cell_points0 + np.array([[r + b, +a]] * 6)

        for xcell_ix in range(Lx):
            for ycell_ix in range(Ly):
                cell_ix = xcell_ix + ycell_ix * Ly
                shift_vector = np.array([3 * (r * xcell_ix + b * ycell_ix), 3 * a * ycell_ix])

                # add edges
                tile_edges = tile_edges0 + np.array([[shift_vector] * 2] * 9)
                ax.plot(*tile_edges.T, edge_color, alpha=edge_alpha, lw=edge_width)

                # add cells
                red_hex = Polygon(red_cell_points0 + np.array([shift_vector] * 6), facecolor=cell_color_red,
                                  alpha=cell_alpha)
                green_hex = Polygon(green_cell_points0 + np.array([shift_vector] * 6), facecolor=cell_color_green,
                                    alpha=cell_alpha)
                blue_hex = Polygon(blue_cell_points0 + np.array([shift_vector] * 6), facecolor=cell_color_blue,
                                   alpha=cell_alpha)

                ax.add_patch(red_hex)
                ax.add_patch(green_hex)
                ax.add_patch(blue_hex)

                if add_cell_labels:
                    ax.text(*shift_vector, str(cell_ix), color=cell_labels_color, alpha=cell_labels_alpha,
                            size=cell_labels_size,
                            horizontalalignment='center', verticalalignment='center')

        if axis == None:
            return fig, ax
