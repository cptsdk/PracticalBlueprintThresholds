# define data_type bool

extern "C" {
    void LossDecoder_GaussElimin(data_type*, int, int, int);
    void LossDecoder_GaussElimin_print(data_type*, int, int, int);
    void SwitchRows(data_type*, int, int, int, int);
    void SubtractRows(data_type*, int, int, int, int);

    void LossDecoder_GaussElimin_trackqbts(data_type*, int*, int, int, int);
    void LossDecoder_GaussElimin_trackqbts_noorderedlost(data_type* mat, int* qbt_syndr_mat, int n_rows, int n_cols, int* lost_qbts, int num_lost_qbts);
    void SwitchRows_trackqbts(data_type*, int*, int, int, int, int);
    void SubtractRows_trackqbts(data_type*, int*, int, int, int, int);

    void PrintMatrix_toTerminal(data_type*, int, int);
    void PrintMatrix_int_toTerminal(int*, int, int);
}
