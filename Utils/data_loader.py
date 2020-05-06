import pandas as pd
import os
import math
import numpy as np
import logging
from Utils.data_structures import PartialObservabilityProblem
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug

def load_dataset_pop(pop: PartialObservabilityProblem, dtype=np.float32, null_dataset=False):
    """
    :type pop: PartialObservabilityProblem
    """
    return load_dataset(pop.dataset_path, pop.test_fraction, pop.num_examples_to_generate, pop.T,
                        pop.hidden_voltage_bus_id_list, pop.hidden_power_bus_id_list,
                        pop.target_voltage_bus_id_list, pop.target_power_bus_id_list,
                        use_polar=False,
                        null_dataset=null_dataset,
                        num_hidden_measurements_to_keep=pop.num_hidden_measurements-pop.num_target_measurements,
                        dtype=dtype)

def load_dataset(dataset_path, test_fraction, num_examples_to_generate, T, hidden_voltage_bus_id_list,
                hidden_power_bus_id_list, target_voltage_bus_id_list, target_power_bus_id_list, use_polar=False,
                 null_dataset=False, restrict_dataset=None, num_hidden_measurements_to_keep=0, dtype=np.float32):
    """
    Reads the CSV files that contain time series of the complex power and voltage measurements of a grid.
    Constructs T-timesteps-long train and test examples, where the last time step provides partial observability.
    The output training set consists of 3 matrices (for training and 3, matrices for test) denoted X1, X2, y
    such that the X1 contains the full observability for the first T-1 time steps. X2 contains the partially observable
    measurements of the Tth step and y contains the hidden measurements of the Tth timestep. Together, all the X1,X2,y
    ndarrays contain N*T*nnodes complex measurements. Or, precisely N*T*nnodes*2 real measurements.

    It is made sure that there is no data leakage between the training series and the test series.
    :param dataset_path: <str> a full path to a directory containing the 4 csv files of the dataset recordings.
    :param num_examples_to_generate - total number of examples to generate from the raw time series.
    :param test_fraction - the fraction of the samples that will be taken aside as "test examples"
    :param T - time series length (each generating example will contain this amount of time steps)
               must be a value between 2 and 100.
    :param null_dataset - set to true if you want a dataset of all zeros.
    :param restrict_dataset - integer number of dataset examples to be considered (prior to splitting)
                              set to None if no restriction should be applied
    :num_hidden_measurements_to_keep - number of hidden measurements in the last time step to include in the X2hidden outputs.
    :dtype - numpy data type of the numbers to be set for all the returned matrices
    :return: 9 ndarrays in the following order:
             X1_train, X2_train, X2hidden_train, y_train <--- ndarrays of shapes (N,T-1,16)    (N,6)     (N,10)     (N,8)
             X1_test, X2_test, X2hidden_test, y_test<--- ndarrays of shapes (N,T-1,16)    (N,6)    (N,10)      (N,8)
             Yreal, Yimag <-- ndarrays of the shape (n_buses, n_buses) each
    """
    Tmin = 2
    Tmax = 100
    if num_examples_to_generate <= T:
        ld("Dataset creation error: the number of examples "
           "to generate (given {}) cannot be smaller"
           " than a single sequence length "
           "(given {}).".format(num_examples_to_generate,T))
        os.sys.exit(-1)
    if 2 > T or T > 100:
        ld("Dataset creation error: the T parameter must be between {}"
           "and {} (given {}).".format(Tmin, Tmax,T))
        os.sys.exit(-1)

    power_real_vec  = pd.read_csv(os.path.join(dataset_path, 'real_S.csv'), header=None, dtype=dtype).values.astype(dtype)
    power_imag_vec  = pd.read_csv(os.path.join(dataset_path, 'imag_S.csv'), header=None, dtype=dtype).values.astype(dtype)
    voltage_mag_vec = pd.read_csv(os.path.join(dataset_path, 'mag_V.csv'), header=None, dtype=dtype).values.astype(dtype)
    voltage_ang_vec = pd.read_csv(os.path.join(dataset_path, 'ang_V.csv'), header=None, dtype=dtype).values.astype(dtype)
    Yreal = pd.read_csv(os.path.join(dataset_path, 'real_Y.csv'), header=None, dtype=dtype).values.astype(dtype)
    Yimag = pd.read_csv(os.path.join(dataset_path, 'imag_Y.csv'), header=None, dtype=dtype).values.astype(dtype)

    power_real_vec   *= (1-int(null_dataset))
    power_imag_vec   *= (1-int(null_dataset))
    voltage_mag_vec  *= (1-int(null_dataset))
    voltage_ang_vec  *= (1-int(null_dataset))

    # Optionally transform to Polar complex form:
    if use_polar:
        power_complex = power_real_vec + 1j * power_imag_vec
        power_vec1 = np.absolute(power_complex)
        power_vec2 = np.angle(power_complex, deg=True)
        voltage_vec1 = voltage_mag_vec
        voltage_vec2 = voltage_ang_vec
        ld("Warning: using polar coordiantes for the dataset may "
           "result in the incorrect power flow equation solving.")
    else:
        power_vec1 = power_real_vec
        power_vec2 = power_imag_vec
        voltage_vec1 = np.multiply(voltage_mag_vec, np.cos(voltage_ang_vec*np.pi/180.0))
        voltage_vec2 = np.multiply(voltage_mag_vec, np.sin(voltage_ang_vec*np.pi/180.0))

    assert (power_vec1.shape == power_vec2.shape == voltage_vec1.shape == voltage_vec2.shape)

    # Optionally throw away some of the data samples
    if restrict_dataset is not None:
        if power_vec1.shape[0] < restrict_dataset:
            ld("Couldn't restrict the dataset (which "
                  "contains {} examples) to {} examples. "
                  "Using the entire dataset.".format(power_vec1.shape[0], restrict_dataset))
        else:
            power_vec1 = power_vec1[:restrict_dataset]
            power_vec2 = power_vec2[:restrict_dataset]
            voltage_vec1 = voltage_vec1[:restrict_dataset]
            voltage_vec2 = voltage_vec2[:restrict_dataset]
            ld("Restricted the dataset from"
                  "{} examples to {} examples. ".format(power_vec1.shape[0], restrict_dataset))



    # split to train and test
    eight_vector_list = train_test_split(power_vec1, power_vec2, voltage_vec1,voltage_vec2,
                                                                 test_size=test_fraction,
                                                                 shuffle=False)
    [pvec1_tr, pvec1_te, pvec2_tr, pvec2_te, vvec1_tr, vvec1_te, vvec2_tr, vvec2_te] = eight_vector_list

    # Construct the X1, X2, y. with proper shapes
    X1_train, X2_train, X2hidden_train, y_train, time_steps_train = createTlongTimeseries(pvec1_tr, pvec2_tr, vvec1_tr, vvec2_tr, int(math.ceil((1-test_fraction)*num_examples_to_generate)), T, Tmax,
                                                                                          hidden_voltage_bus_id_list, hidden_power_bus_id_list, target_voltage_bus_id_list, target_power_bus_id_list, dtype, randomized_timestep_order=True)
    X1_test, X2_test, X2hidden_test, y_test, time_steps_test = createTlongTimeseries(pvec1_te, pvec2_te, vvec1_te, vvec2_te, int(math.floor(test_fraction*num_examples_to_generate)), T, Tmax,
                                                                                     hidden_voltage_bus_id_list, hidden_power_bus_id_list, target_voltage_bus_id_list, target_power_bus_id_list, dtype, randomized_timestep_order=False)
    return X1_train, X2_train, X2hidden_train[:,:num_hidden_measurements_to_keep], y_train, time_steps_train, X1_test, X2_test, X2hidden_test[:,:num_hidden_measurements_to_keep], y_test, Yreal, Yimag, time_steps_test

def createTlongTimeseries(power_vec1, power_vec2, voltage_vec1, voltage_vec2, num_examples_to_generate, T, Tmax,
                         hidden_voltage_bus_id_list, hidden_power_bus_id_list, target_voltage_bus_id_list, target_power_bus_id_list,
                          dtype=np.float32, randomized_timestep_order=True):
    """
    Receives 2 vectors for power and 2 vectors for voltage, creates X1, X2, X2hidden, y for training / testing

    """
    # Specific dataset definitions:
    n_pmu_measurements_per_bus = 4  # Re(V), Im(V), Re(S), Im(S)
    num_buses_in_dataset = power_vec1.shape[1]
    num_all_measurements = n_pmu_measurements_per_bus * num_buses_in_dataset
    num_hidden_measurements = (len(hidden_power_bus_id_list) + len(hidden_voltage_bus_id_list)) * 2
    num_target_measurements = (len(target_voltage_bus_id_list) + len(target_power_bus_id_list)) * 2
    num_remaining_measurements = num_all_measurements - num_hidden_measurements
    X1 = np.zeros((num_examples_to_generate, T - 1, num_all_measurements), dtype=dtype)  # time steps 0,...,T-2 will contain all the measurements.
    X2 = np.zeros((num_examples_to_generate, num_remaining_measurements), dtype=dtype)  # The auxilliary input at time T-1 will contain only 3 power phasors.
    X2hidden = np.zeros((num_examples_to_generate, num_hidden_measurements), dtype=dtype)  # The hidden input at time T-1 will contain the missing 1st power phasor.
    y = np.zeros((num_examples_to_generate, num_target_measurements), dtype=dtype)
    #random_interval_begins = np.random.choice(power_vec1.shape[0]-T, num_examples_to_generate)
    if randomized_timestep_order:
        random_interval_endpoints = Tmax + np.random.choice(power_vec1.shape[0] - Tmax, num_examples_to_generate)
    else:
        random_interval_endpoints = np.linspace(Tmax,power_vec1.shape[0], num_examples_to_generate, dtype=np.int32)
    for i in range(num_examples_to_generate):

        # Re arrange the power_vec1, power_vec2, voltage_vec1, voltage_vec2
        # arrays into dataset of containing the ndarrays: X1, X2, y.
        #
        # Example for the 4-node grid
        # The i^th dataset example in the following shape:
        # X1 shape:((T-1)x16):
        #  t=0   Re(S1), Im(S1), ..., Re(S4), Im(S4) |  Re(V1), Im(V1),...,Re(V4), Im(V4)
        #  t=1   Re(S1), Im(S1), ..., Re(S4), Im(S4) |  Re(V1), Im(V1),...,Re(V4), Im(V4)
        #  ...
        #  t=T-2 Re(S1), Im(S1), ..., Re(S4), Im(S4) |  Re(V1), Im(V1),...,Re(V4), Im(V4)
        #
        # X2 shape:(6):
        #  Re(S2), Im(S2), Re(S3), Im(S3), Re(S4), Im(S4)   <-- from t=T-1
        #
        # X2hidden shape: (10)
        #  Re(S1), Im(S1), Re(V1), Im(V1), Re(V2), Im(V2), Re(V3), Im(V3), Re(V4), Im(V4)  <-- from t=T-1
        #
        # y shape:(8):
        #  Re(V1), Im(V1), Re(V2), Im(V2), Re(V3), Im(V3), Re(V4), Im(V4)   <-- from t=T-1
        ##
        # t_start = random_interval_begins[i]
        # t_end = t_start + T
        t_end = random_interval_endpoints[i]
        t_start = t_end - T
        arrid_x2_p = 0
        arrid_x2_v = 2 * (num_buses_in_dataset - len(hidden_power_bus_id_list))
        arrid_x2hidden_p = 0
        arrid_x2hidden_v = 2 * len(hidden_power_bus_id_list)
        arrid_y_p = 0
        arrid_y_v = 2 * len(target_power_bus_id_list)
        for busid in range(num_buses_in_dataset):

            # Construct X1
            #pdb.set_trace()
            X1[i, :, 2 * busid] = power_vec1[t_start:t_end - 1, busid]
            X1[i, :, 2 * busid + 1] = power_vec2[t_start:t_end - 1, busid]
            X1[i, :, 2 * busid + n_pmu_measurements_per_bus * num_buses_in_dataset // 2] = voltage_vec1[
                                                                                          t_start:t_end - 1, busid]
            X1[i, :, 2 * busid + 1 + n_pmu_measurements_per_bus * num_buses_in_dataset // 2] = voltage_vec2[
                                                                                              t_start:t_end - 1, busid]

            # Construct X2
            if busid not in hidden_power_bus_id_list:
                X2[i, arrid_x2_p] = power_vec1[t_end - 1, busid]
                X2[i, arrid_x2_p + 1] = power_vec2[t_end - 1, busid]
                arrid_x2_p += 2
            else:
                X2hidden[i, arrid_x2hidden_p] = power_vec1[t_end - 1, busid]
                X2hidden[i, arrid_x2hidden_p + 1] = power_vec2[t_end - 1, busid]
                arrid_x2hidden_p += 2
            if busid not in hidden_voltage_bus_id_list:
                X2[i, arrid_x2_v] = voltage_vec1[t_end - 1, busid]
                X2[i, arrid_x2_v + 1] = voltage_vec2[t_end - 1, busid]
                arrid_x2_v += 2
            else:
                X2hidden[i, arrid_x2hidden_v] = voltage_vec1[t_end - 1, busid]
                X2hidden[i, arrid_x2hidden_v + 1] = voltage_vec2[t_end - 1, busid]
                arrid_x2hidden_v += 2
            # Construct y
            if busid in target_power_bus_id_list:
                y[i, arrid_y_p] = power_vec1[t_end - 1, busid]
                y[i, arrid_y_p + 1] = power_vec2[t_end - 1, busid]
                arrid_y_p += 2
            if busid in target_voltage_bus_id_list:
                y[i, arrid_y_v] = voltage_vec1[t_end - 1, busid]
                y[i, arrid_y_v + 1] = voltage_vec2[t_end - 1, busid]
                arrid_y_v += 2
    return X1, X2, X2hidden, y, random_interval_endpoints