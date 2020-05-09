import argparse

import numpy as np
from Utils.data_loader import load_dataset, load_dataset_pop
import random

from Utils.data_structures import create_partial_observability_problem


def calc_error(power_vector, voltage_vector, Y_real_np, Y_imag_np):
    """ Computes the mean squared error between the right-hand-side  and the
        left  hand side of the power flow system of equations.
        The variable values used in the solution of PFE are taken from the ""example_id""th example
        in the provided data-sets (X2hidden_train, X2_train, y_train) and with respect to the admittance
        matrix ormed by Y_real_np, Y_imag_np

        :param power_vector - a vector containing interleaved real and imaginary values of power for each node.
                              all must respect a cartesian representation of complex numbers.
                              for example:
                              [Re(S0), Im(S0), Re(S1), Im(S1), ..., Re(Sn), Im(Sn)]
        :param voltage_vector - a vector containing interleaved real and imaginary values of voltage for each node.
                              all must respect a cartesian representation of complex numbers.
                              for example:
                              [Re(V0), Im(V0), Re(V1), Im(V1), ..., Re(Vn), Im(Vn)]
        :param Y_real_np - an n-by-n real valued matrix representing the conductance matrix.
        :param Y_imag_np - an n-by-n real valued matrix representing the susdeptance matrix matrix.

        The aim is to find the numerical error when allegedly a valid solution is fed to this function.
    """
    assert(len(Y_real_np.shape)==len(Y_imag_np.shape)==2*len(voltage_vector.shape)==2*len(power_vector.shape))
    assert(2*Y_real_np.shape[0]==2*Y_real_np.shape[1]==2*Y_imag_np.shape[0]==2*Y_imag_np.shape[1]==power_vector.shape[0]==voltage_vector.shape[0])
    n_buses = Y_real_np.shape[0]
    n_target_measurements = n_buses * 2 # two target measurements per bus (` real measurement and 1 imaginary measurement)
    real_indices = np.array(range(0, n_target_measurements - 1, 2))
    imag_indices = np.array(range(1, n_target_measurements, 2))
    Y = Y_real_np + 1j * Y_imag_np
    sr = np.take(power_vector, real_indices)
    si = np.take(power_vector, imag_indices)
    s = sr + 1j * si
    vr = np.take(voltage_vector, real_indices)
    vi = np.take(voltage_vector, imag_indices)
    v = vr + 1j * vi
    rhs = np.matmul(np.matmul(np.diag(v), np.conj(Y)), np.conj(v))
    e_cplx = s - rhs
    e = np.sum((np.square(np.real(e_cplx)) + np.square(np.imag(e_cplx)))) / (2*e_cplx.shape[0])
    return e


def main(args):
    np.random.seed(42)
    random.seed(42)
    dataset_name = args.dataset_name
    dataset_dir = 'Dataset'
    nTrain_examples_to_evaluate = args.nTrain
    nTest_examples_to_evaluate = args.nTest
    pop = create_partial_observability_problem(dataset_dir, dataset_name, T=5, Ns=36, Nv=0,
                                               verbose=True, reverse_bus_hiding_order=False)

    paked_dataset = load_dataset_pop(pop, dtype=np.float64, null_dataset=False)
    X2_train, X2hidden_train, y_train = paked_dataset[1], paked_dataset[2], paked_dataset[3]
    X2_test, X2hidden_test, y_test = paked_dataset[6], paked_dataset[7], paked_dataset[8]
    Y_real_np, Y_imag_np = paked_dataset[9], paked_dataset[10]

    print("Showing Power-Flow_Equation (PFE) error, that is the mean square error between the "
          "equation sides of the equation) for  of a "
          "dataset at the path:\n{}".format(dataset_dir))
    print("---Training Set (first {} examples) ---".format(nTrain_examples_to_evaluate))
    train_maxe = 0.0
    train_sume = 0.0

    for example_id in range(nTrain_examples_to_evaluate):
        power_vector, voltage_vector = np.concatenate([X2_train[example_id], X2hidden_train[example_id]]), y_train[example_id]
        e=calc_error(power_vector, voltage_vector, Y_real_np, Y_imag_np)
        train_maxe = max(train_maxe,e)
        train_sume += e
        print("  Train Example {} - PFE error is {:.3E}".format(example_id, e))

    print("---Test Set (first {} examples) ---".format(nTest_examples_to_evaluate))
    test_maxe = 0.0
    test_sume = 0.0
    for example_id in range(nTest_examples_to_evaluate):
        power_vector, voltage_vector = np.concatenate([X2_test[example_id], X2hidden_test[example_id]]), y_test[example_id]
        e=calc_error(power_vector, voltage_vector, Y_real_np, Y_imag_np)
        test_maxe = max(test_maxe, e)
        test_sume += e
        print("  Test Example {} - PFE error is {:.3E}".format(example_id, e))

    print("---Summary---")
    print(" Dataset at {}/{}".format(dataset_dir, dataset_name))
    print(" Examined first {} training examples and first {} test examples)".format(nTrain_examples_to_evaluate,
                                                                                    nTest_examples_to_evaluate))
    print(" Training set average PFE error: {:.3E}".format(train_sume/nTrain_examples_to_evaluate))
    print(" Training set highest PFE error: {:.3E}".format(train_maxe))
    print(" Test set average PFE error: {:.3E}".format(test_sume/nTest_examples_to_evaluate))
    print(" Test set highest PFE error: {:.3E}".format(test_maxe))

if __name__ == "__main__":
    """
    This program validates that the dataset values (inputs and targets, jointly) at 
    each time-step respect the powerflow equations (PFE). Numerical error values, which 
    stand for the mismatch between the power and the voltage values in the dataset, are reported.
    Note: as the generation of the dataset is probabilistic, this script checks only one realization
    of the dataset. Yet, this validation process should provide a good estimation of the general 
    PFE numeric error, inherent in the dataset.
    """
    parser = argparse.ArgumentParser(description="This program validates that the dataset values (inputs "
                                                 "and targets, jointly) at each time-step respect the powerflow "
                                                 "equations (PFE). Numerical error values, which  stand for the "
                                                 "mismatch between the power and the voltage values in the dataset, "
                                                 "are reported. Note: as the generation of the dataset is probabilistic, "
                                                 "this script checks only one realization of the dataset. Yet, this "
                                                 "validation process should provide a good estimation of the general "
                                                 "PFE numeric error, inherent in the dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model-name', type=str,
                        default='ieee37_smooth_ord_60_downsampling_factor_60',
                        help='Dataset name to be examined')
    parser.add_argument('-nTrain', type=int,
                        default=8100,
                        help='number of training examples to be examined for numerical errors')
    parser.add_argument('-nTest', type=int,
                        default=900,
                        help='number of training examples to be examined for numerical errors')
    args = parser.parse_args()
    main(args)



