import numpy as np
from Utils.data_loader import load_dataset
import random


def calc_error(X2hidden_train, X2_train, y_train, Y_real_np, Y_imag_np, example_id):

    num_target_measurements = 8
    real_indices = np.array(range(0, num_target_measurements - 1, 2))
    imag_indices = np.array(range(1, num_target_measurements, 2))
    Y = Y_real_np + 1j * Y_imag_np
    sconcat = np.concatenate([X2hidden_train[example_id], X2_train[example_id]])
    sr = np.take(sconcat, real_indices)
    si = np.take(sconcat, imag_indices)
    s = sr + 1j * si
    vr = np.take(y_train[example_id], real_indices)
    vi = np.take(y_train[example_id], imag_indices)
    v = vr + 1j * vi
    rhs = np.matmul(np.matmul(np.diag(v), np.conj(Y)), np.conj(v))
    e_cplx = s - rhs
    e = np.sum((np.square(np.real(e_cplx)) + np.square(np.imag(e_cplx)))) / (2*e_cplx.shape[0])
    return e


def main():
    np.random.seed(42)
    random.seed(42)
    dataset_name = 'solar_smooth_ord_60_downsampling_factor_60'
    dataset_dir = 'Dataset'
    test_fraction = 0.1
    num_examples_to_generate = 9000
    T=100
    num_hidden_measurements_to_keep=2
    X1_train, X2_train, X2hidden_train, y_train, X1_test, X2_test, X2hidden_test, y_test, Y_real_np, Y_imag_np = load_dataset(
        dataset_dir,
        dataset_name,
        test_fraction,
        num_examples_to_generate,
        T=T,
        num_hidden_measurements_to_keep=num_hidden_measurements_to_keep)
    for example_id in range(20):
        e=calc_error(X2hidden_train, X2_train, y_train, Y_real_np, Y_imag_np, example_id)
        print("Error is {}".format(e))

if __name__ == "__main__":
    """
    This program validates that the dataset values (inputs and targets, jointly) at each time-step respect the powerflow equations.
    """
    main()



