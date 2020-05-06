import os
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug


class PartialObservabilityProblem:
    def __init__(self,
                 dataset_path,
                 num_nodes,
                 T,
                 num_examples_to_generate,
                 test_fraction,
                 validation_fraction,
                 hidden_power_bus_id_list,
                 hidden_voltage_bus_id_list,
                 target_power_bus_id_list,
                 target_voltage_bus_id_list,
                 reference_bus_id,
                 Ns,
                 Nv):
        """
        A data structure that captures the complete description of the partial observability
        problem in power grids.
        :param dataset_path: a string specifying the path to the dataset containing the
                             power and voltage recordings (4 csv files).
        :param num_nodes: number of buses in the grid
        :param T: number of time-steps in the scope of a single train/test example.
                  The last time step is partially observable
        :param num_examples_to_generate: number of examples to be drawn at random from the
                                         power-voltage records.
        :param test_fraction: fractional number in  [0.0,1.0]. Out of the dataset provided,
                              which fraction of the examples will be put aside as a test
                              set.
        :param validation_fraction: fractional number in  [0.0,1.0]. Out of the training set
                               provided, which fraction of the examples will be put aside as
                               a validation set.
        :param hidden_power_bus_id_list:  a list of bus ids, whose power is not observable at
                                          the last time step.
        :param hidden_voltage_bus_id_list:a list of bus ids, whose voltage is not observable at
                                          the last time step.
        :param target_power_bus_id_list:  a list of bus ids, whose power is to be predicted
                                          the last time step.
        :param target_voltage_bus_id_list:a list of bus ids, whose voltage is to be predicted
                                          the last time step.
        :param reference_bus_id: an integer, id of the bus defined as a "slack" or aa reference bus.
        :param Ns: Number of observable power measurements in the last time step
        :param Nv: Number of observable voltage measurements in the last time step
        """

        # TODO: add slack bus id as a data member.

        self.dataset_path = dataset_path
        self.num_nodes = num_nodes
        self.T = T
        self.num_examples_to_generate = num_examples_to_generate
        self.test_fraction = test_fraction
        self.validation_fraction = validation_fraction
        self.hidden_power_bus_id_list = hidden_power_bus_id_list
        self.hidden_voltage_bus_id_list = hidden_voltage_bus_id_list
        self.visible_power_bus_id_list = list(sorted(set(range(num_nodes)) - set(hidden_power_bus_id_list)))# complementary to the hidden one
        self.visible_voltage_bus_id_list = list(sorted(set(range(num_nodes)) - set(hidden_voltage_bus_id_list)))# complementary to the hidden one
        self.target_power_bus_id_list = target_power_bus_id_list
        self.target_voltage_bus_id_list = target_voltage_bus_id_list
        self.reference_bus_id = reference_bus_id
        self.Ns = Ns
        self.Nv = Nv

        # Measurement counts
        self.num_phasors_per_bus = 2 # there are 2 phasors per bus (S,V)
        self.num_measurements_per_phasor = 2 # phasors are complex (2 values)
        self.num_hidden_power_measurements = len(self.hidden_power_bus_id_list) * self.num_measurements_per_phasor
        self.num_hidden_voltage_measurements = len(self.hidden_voltage_bus_id_list) * self.num_measurements_per_phasor
        self.num_hidden_measurements = self.num_hidden_power_measurements + self.num_hidden_voltage_measurements
        self.num_target_power_measurements = len(self.target_power_bus_id_list) * self.num_measurements_per_phasor
        self.num_target_voltage_measurements = len(self.target_voltage_bus_id_list) * self.num_measurements_per_phasor
        self.num_target_measurements = self.num_target_voltage_measurements + self.num_target_power_measurements
        self.num_target_buses = self.num_target_measurements // self.num_phasors_per_bus if self.num_target_power_measurements == 0 else -1 # this value will be available only when no power measurements are seeked.
        self.num_visible_power_measurements = len(self.visible_power_bus_id_list) * self.num_measurements_per_phasor
        self.num_visible_voltage_measurements = len(self.visible_voltage_bus_id_list) * self.num_measurements_per_phasor
        self.num_all_measurements = self.num_nodes * self.num_phasors_per_bus * self.num_measurements_per_phasor
        self.num_remaining_measurements = self.num_all_measurements - self.num_hidden_measurements # what is left after all the hidden measurements are removed.

        assert(Ns * self.num_measurements_per_phasor == self.num_visible_power_measurements)
        assert(Nv * self.num_measurements_per_phasor == self.num_visible_voltage_measurements)
        assert(self.num_all_measurements == self.num_visible_voltage_measurements + self.num_visible_power_measurements + self.num_hidden_voltage_measurements + self.num_hidden_power_measurements)

def set_hidden_measurement_lists_from_Ns_Nv(num_nodes, Ns, Nv, list_bus_id_power_hiding_priority=None, list_bus_id_voltage_hiding_priority=None):
    """
        Returns the list of the hidden power bus ids and a list of hidden voltage ids
        :param num_nodes: number of buses in the grid
        :param Ns: Number of observable power measurements in the last time step
        :param Nv: Number of observable voltage measurements in the last time step
        :param list_bus_id_power_hiding_priority: list of bus indices which was sorted according to the preferred
                    order of hiding. Index 0 of this list corresponds to the most likely bus to be hidden.
        :param list_bus_id_voltage_hiding_priority: list of bus indices which was sorted according to the preferred
        order of hiding. Index 0 of this list corresponds to the most likely bus to be hidden.
        :return:
        """
    if list_bus_id_power_hiding_priority is None:
        list_bus_id_power_hiding_priority = list(range(num_nodes))
    if list_bus_id_voltage_hiding_priority is None:
        list_bus_id_voltage_hiding_priority = list(range(num_nodes))
    hidden_power_bus_id_list = []
    next_busid_to_hide = 0
    for bus_id in range(Ns, num_nodes):
        hidden_power_bus_id_list.append(list_bus_id_power_hiding_priority[next_busid_to_hide])
        next_busid_to_hide += 1

    hidden_voltage_bus_id_list = []
    next_busid_to_hide = 0
    for bus_id in range(Nv, num_nodes):
        hidden_voltage_bus_id_list.append(list_bus_id_voltage_hiding_priority[next_busid_to_hide])
        next_busid_to_hide += 1
    hidden_power_bus_id_list.sort()
    hidden_voltage_bus_id_list.sort()
    return hidden_power_bus_id_list, hidden_voltage_bus_id_list


def set_hidden_measurement_lists_from_observability(num_nodes, observability, list_bus_id_hiding_priority=None):
    """
    Returns the list of the hidden power bus ids and a list of hidden voltage ids
    :param num_nodes: number of buses in the grid
    :param observability: a fractional number in [0.0, 1.0] which
                        sets the observability degree considered
                        in the problem.
    :param list_bus_id_hiding_priority: list of bus indices which was sorted according to the preferred
                order of hiding. Index 0 of this list corresponds to the most likely bus to be hidden.
    :return:
    """
    if list_bus_id_hiding_priority is None:
        list_bus_id_hiding_priority = list(range(num_nodes))
    observability_step_size = 1 / float(2 * num_nodes)
    hidden_power_bus_id_list = []
    next_busid_to_hide = 0
    for observability_step in range(1,num_nodes+1):
        threshold_for_current_measurement = observability_step * observability_step_size
        if threshold_for_current_measurement >= observability:
            hidden_power_bus_id_list.append(list_bus_id_hiding_priority[next_busid_to_hide])
            next_busid_to_hide += 1

    hidden_voltage_bus_id_list = []
    next_busid_to_hide = 0
    for observability_step in range(1,num_nodes+1):
        threshold_for_current_measurement = 0.5 + observability_step * observability_step_size
        if threshold_for_current_measurement >= observability:
            hidden_voltage_bus_id_list.append(list_bus_id_hiding_priority[next_busid_to_hide])
            next_busid_to_hide += 1

    hidden_power_bus_id_list.sort()
    hidden_voltage_bus_id_list.sort()
    return hidden_power_bus_id_list, hidden_voltage_bus_id_list


def make_str_for_pretty_print_int_list(lst):
    """
    Produce a stirng which neatly prints a list of integers.
    This is done by compacting the integers into contiguous ranges.
    for example [0,1,2,3,4,10,11,12] will become "[0..4,10..12]"
    :param lst: list of integers
    :return: string
    """
    stri="["
    prev=None
    seq_start_num = None
    for i,num in enumerate(lst):
        if prev is None:
            # Warmup
            seq_start_num = num
            stri = stri + str(num)
        elif prev != num - 1:
                if seq_start_num != prev:
                    # Previous sequence contained more than 1 number.
                    if seq_start_num == prev-1:
                        stri = stri + ", " + str(prev)
                    else:
                        stri = stri + ".." + str(prev)

                # Start new sequence

                stri = stri + ", " + str(num)
                seq_start_num = num
        elif i==len(lst)-1:
            if seq_start_num != num:
                # Previous sequence contained more than 1 number.
                if seq_start_num == prev:
                    stri = stri + ", " + str(num)
                else:
                    stri = stri + ".." + str(num)
        prev = num
    stri = stri +"]"
    return stri

def create_partial_observability_problem(dataset_dir, dataset_name, T, Ns, Nv, verbose=True, reverse_bus_hiding_order=False):
    """
    Constructs a setting of a partial observability problem.
    This function mainly determines the number of nodes and
    sets the concrete bus ids for being hidden, targeted,
    etc. All with accordance to the well known data sets and
    to the observability degree specified as the [0,1]
    fractional parameter "observability".
    :param dataset_dir: a directory that contains all the datasets
    :param dataset_name: a directory name of the dataset
    :param T: Number of time steps to be observed at
    :param Ns: Number of observable power measurements in the last time step
    :param Nv: Number of observable voltage measurements in the last time step
    :param verbose: boolean - if true then upon the creation of the pop
                    object - its attributes will be printed.
    :return:
    """
    # Common setting:
    dataset_path = os.path.join(dataset_dir, dataset_name)

    if dataset_name == 'solar_smooth_ord_60_downsampling_factor_60':
        # 4-nodes grid with 10080 recorded time steps
        num_nodes = 4
        reference_bus_id = 3
        num_examples_to_generate = 9000  # how many examples will be generated from the existing CSV files (generation is carried out via random T-long time series).
        test_fraction = 0.1  # fraction of the generated examples that will become a test set. The splitting between the training and test time series is leakage-safe. Namely, no training time series overlaps with test time series.
        validation_fraction = 0.0  # fraction of the train examples that will become a validation set. Warning: th ecurrent way of splitting the training to validation creates data leakage between the trainin  and validation since the time series overlap!

        # Set the observed bus id lists according to the "observability" parameter in a contiguous manner.
        # TODO: Make sure that custom id list is synchronized with the following processing (in the neural net etc)
        bus_hiding_priority_list = [0, 1, 2, 3]
        bus_hiding_priority_list = list(reversed(bus_hiding_priority_list)) if reverse_bus_hiding_order else bus_hiding_priority_list
        hidden_power_bus_id_list, hidden_voltage_bus_id_list = set_hidden_measurement_lists_from_Ns_Nv(num_nodes, Ns, Nv,
                                                                                                       bus_hiding_priority_list,
                                                                                                       bus_hiding_priority_list)

        # Target bus ids:
        # We assume that we only want to estimate all the voltage and none of the powers
        # as the powers are easily recoverable once the voltages are estimated
        target_power_bus_id_list = []
        target_voltage_bus_id_list = list(range(num_nodes))

        # Example for observability=0.45 in the :
        # hidden_power_bus_id_list = [0]  # hidden from input in T-1 (last) time-step
        # hidden_voltage_bus_id_list = [0, 1, 2, 3]  # hidden from input in T-1 (last) time-step
        # target_power_bus_id_list = []
        # target_voltage_bus_id_list = [0, 1, 2, 3]


    elif dataset_name == 'ieee37_smooth_ord_60_downsampling_factor_60':
        # 36-nodes grid with 10080 recorded time steps
        num_nodes = 36
        reference_bus_id = 0
        num_examples_to_generate = 9000  # how many examples will be generated from the existing CSV files (generation is carried out via random T-long time series).
        test_fraction = 0.1  # fraction of the generated examples that will become a test set. The splitting between the training and test time series is leakage-safe. Namely, no training time series overlaps with test time series.
        validation_fraction = 0.0  # fraction of the train examples that will become a validation set. Warning: th ecurrent way of splitting the training to validation creates data leakage between the trainin  and validation since the time series overlap!

        # Set the observed bus id lists according to the "observability" parameter in a contiguous manner.
        # TODO: Make sure that custom id list is synchronized with the following processing (in the neural net etc)
        bus_hiding_priority_list = list(reversed(range(num_nodes))) # This creates a topological ordering of the nodes, such that the reference bus (slack bus) is the last to be hidden.
        bus_hiding_priority_list = list(reversed(bus_hiding_priority_list[:-1]))+[bus_hiding_priority_list[-1]] if reverse_bus_hiding_order else bus_hiding_priority_list
        hidden_power_bus_id_list, hidden_voltage_bus_id_list = set_hidden_measurement_lists_from_Ns_Nv(num_nodes, Ns, Nv,
                                                                                                       bus_hiding_priority_list,
                                                                                                       bus_hiding_priority_list)

        # Target bus ids:
        # We assume that we only want to estimate all the voltage and none of the powers
        # as the powers are easily recoverable once the voltages are estimated
        target_power_bus_id_list = []
        target_voltage_bus_id_list = list(range(num_nodes))

    else:
        raise NameError("Unknown dataset required \"{}\"".format(dataset_name))


    pop = PartialObservabilityProblem(dataset_path, num_nodes, T, num_examples_to_generate, test_fraction,
                                      validation_fraction, hidden_power_bus_id_list, hidden_voltage_bus_id_list,
                                      target_power_bus_id_list, target_voltage_bus_id_list, reference_bus_id,
                                      Ns, Nv)

    if verbose:
        ld("Created PartialObservabilityProblem scenario:")
        ld(" Dataset name: {}".format(dataset_name))
        ld(" num_nodes: {}".format(num_nodes))
        ld(" T: {}".format(T))
        ld(" (Ns) number of observable bus powers at time=T-1: {}".format(Ns))
        ld(" (Nv) number of observable bus voltages at time=T-1: {}".format(Nv))
        ld(" num_examples_to_generate: {}".format(num_examples_to_generate))
        ld(" test_fraction: {}".format(test_fraction))
        ld(" validation_fraction: {}".format(validation_fraction))
        ld(" hidden_power_bus_id_list: {}".format(make_str_for_pretty_print_int_list(hidden_power_bus_id_list)))
        ld(" hidden_voltage_bus_id_list: {}".format(make_str_for_pretty_print_int_list(hidden_voltage_bus_id_list)))
        ld(" target_power_bus_id_list: {}".format(make_str_for_pretty_print_int_list(target_power_bus_id_list)))
        ld(" target_voltage_bus_id_list: {}".format(make_str_for_pretty_print_int_list(target_voltage_bus_id_list)))



    return pop
