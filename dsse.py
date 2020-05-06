import math
import argparse
import math
import random
import sys
from math import ceil

import progressbar
from matplotlib.markers import MarkerStyle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

from Utils.complex_numbers import realImagVectorsToMagAngVectors
from Utils.data_loader import load_dataset_pop
from Utils.data_structures import create_partial_observability_problem
from Utils.data_transformation import *
from Utils.logprints import *
from Utils.loss import *
from Utils.nn import *
from Utils.plot import plotTwoLines, plotListOfScatters, plotManyLines
from wls import solve_wls
from wls_with_power import solve_wls_with_power


def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x
def restricted_int(x):
    x = int(x)
    if x < 0 or x > 36:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

##################################################################


logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug


parser = argparse.ArgumentParser(description="Distributed system state estimation (DSSE) from partial "
                                             "observability. This script trains and tests a deep neural "
                                             "network (DNN). which receives time series of PMU (Phasor "
                                             "Measurment Unit) measurements such as complex power and complex "
                                             "voltage at the buses (nodes) of a distribution power grid. The "
                                             "DNN receives T-1 time steps of full observability followed by and "
                                             "additional time step with partial observability of the powers and "
                                             "voltages. The aim is to complete the voltage estimation at the Ts "
                                             "timestamp.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-model-name', type=str,
                    default='dsse',
                    help='Model name to be stored')
parser.add_argument('-model-type', choices=['neuralnet', 'persistent', 'wls'],
                    default='neuralnet',
                    help='Chooses whether the neural network model is trained/evaluated in '
                         'this run, or its non-deep-learning alternatives. '
                         '\"persistent\" completely degenerate the neural network '
                         'model into a model that predicts the voltages by the most recently '
                         'observed voltage. \"wls\" sets the model to be weighted least squares'
                         'estimator of the target values. \'neuralnet\" is the default choice that'
                         'sets the neural network as the model type.')
parser.add_argument('--no-training', action="store_true",
                    help='If set - only test will be performed on the model_name '
                         '(not on the model_name_for_loading)')
parser.add_argument('--restore-session', action="store_true",
                    help='A model with the name, specified in the next argument '
                         '(model_name_for_loading), will be loaded and then finetuned.')
parser.add_argument('-model-name-for-loading', type=str,
                    default='dsse',
                    help='Name of model to be loaded to begin the '
                         'training with (aka to be finetuned)')
parser.add_argument('-dataset-name', type=str, choices=['ieee37_smooth_ord_60_downsampling_factor_60', 'solar_smooth_ord_60_downsampling_factor_60'],
                    default='ieee37_smooth_ord_60_downsampling_factor_60',
                    help='Name of the dataset to be loaded.')
parser.add_argument('-data-transform', choices=['normalize', 'standardize', 'none'],
                    default='standardize',
                    help='Choose whether to normalize or to standardize the dataset. Anyway, '
                         'this will only influence the data for the internal use of the model.'
                         'The final results of the regression will appear after inverse transform.')
parser.add_argument('-logdir', type=str,
                    default="Logs",
                    help='Path to a directory where the log files of the runs should be created.')
parser.add_argument('-n-epochs', type=int,
                    default=150,
                    help='number of epochs the training will run')
parser.add_argument('-batch-size', type=int,
                    default=50,
                    help='number of training examples that constitute '
                         'a single minibatch.')
parser.add_argument('-num-examples-to-generate', type=int,
                    default=9000,
                    help='number of dataset time-series-examples to be produced '
                         'from the original dataset examples.')
parser.add_argument('-gpuid', type=int,
                    default=0,
                    help='the id of the GPU to host the run.')
parser.add_argument('-weight-decay', type=float,
                    default=0.0,
                    help='L2 norm penaly will be multiplied by this coefficient '
                         'prior to be added to loss function. ')
parser.add_argument('-equations-regularizer', type=float,
                    default=1.0,
                    help='The deviation of the estimated voltages from feasible '
                         'power flow equations solution will be multiplied by this '
                         'coefficient prior to be added to loss function.')
parser.add_argument('-T', type=int,
                    default=50,
                    help='Number of PMU measurment time-steps in a single input sequence.')
parser.add_argument('-Ns', type=restricted_int,
                    default=35,
                    help='Number of observable power measurements in the last time step')
parser.add_argument('-Nv', type=restricted_int,
                    default=0,
                    help='Number of observable voltage measurements in the last time step')
parser.add_argument('-n-layers-lstm', type=int,
                    default=2,
                    help='Number of LSTM layers stacked')
parser.add_argument('-n-layers-fc', type=int,
                    default=5,
                    help='number of fully connected layers, which follow the LSTM layers.')
parser.add_argument('-n-layers-fc-partially-obs', type=int,
                    default=2,
                    help='number of fully connected layers, which process the partially observable measurements.')
parser.add_argument('--use-complex-tf-ops', action="store_true",
                    help='If used, then complex data types will be employed in the computation. '
                         'Warning: this was found as unreliable whenever gradient computations '
                         'were required.')
parser.add_argument('--tensorboard', action="store_true",
                    help='turns on tensorboard logging"')
parser.add_argument('--null-dataset', action="store_true",
                    help='If used, this option zeros all the data examples (train and test). '
                         'This is useful for monkey testing of the neural net.')
parser.add_argument('--revert-to-single-precision', action="store_true",
                    help='If used, all the computations beginning from the dataset preprocessing,'
                         'and all the way to the neural net and postprocessing - will be done using'
                         'single precision instead of the default double precision.')
parser.add_argument('--wls-with-power', action="store_true",
                    help='If used, then the N-Ns unobservable powers will be attached to the N unobserved' \
                         'voltages and serve as optimization variables in the WLS. Otherwise, the N-Ns ' \
                         'unobsevable powers will be only guessed but not optimized. ')
parser.add_argument('--wls-weights-disabled', action="store_true",
                    help='If used, then if model-type is wls, the weights are degenerated '
                         'to be all ones')
parser.add_argument('--wls-weights-discriminate-hidden', action="store_true",
                    help='If used, then if model-type is wls, the weights for all '
                         'the power phasors are set inversely proportional to their own'
                         'standard deviations. This is instead of setting the weights only for the '
                         'hidden variables according to the std, and setting the visible powers '
                         'weights 10 times larger than the largest weight of the hidden measurement')
parser.add_argument('--no-trained-scaling', action="store_true",
                    help='If used, then scaling (multiplicative and additive) of the logits will be '
                         'disabled (At the end of the model).')
parser.add_argument('--no-prediction-plots', action="store_true",
                    help='If used, then no per-test-example plots will be generated during the final'
                         'model testing. Only the chronological plot wil be generated in the '
                         'predictions directory')
parser.add_argument('--reverse-bus-hiding-order', action="store_true",
                    help='If used, then the Ns or Nv buses will be taken with respect to a reversed'
                         'order of buses')
parser.add_argument('--slim-auxilliary-network', action="store_true",
                    help='If used, then a slimmer sub-model (Ns-->N/6-->N/6) in charge of '
                         't-1 input (partially observable) is used instead of a Ns-->Ns-->N/6.')
parser.add_argument('-seed', type=int, default=42,
                    help='This number sets the initial pseudo-random generators of the numpy '
                         'and of the tensorflow (default:42)')


args = parser.parse_args()

# Main toggles
model_name                  = args.model_name
dataset_dir                 = "Dataset"
dataset_name                = args.dataset_name
restore_session             = args.restore_session         # if True, a previously saved model will be restored PRIOR to training
model_name_for_loading      = args.model_name_for_loading # specify the model-name that should be loaded.
random_seed                 = args.seed
figures_outputdir           = "Figures"
log_outputdir               = args.logdir
tensorboardDir              = "TensorBoardLogs"
TensorboardEnabled_valid    = False
TensorboardEnabled_training = args.tensorboard
tfcfg                       = tf.ConfigProto()
tfcfg.gpu_options.allow_growth = True          #tf.ConfigProto(device_count = {'GPU':0}) #tf.ConfigProto(log_device_placement=True)
tfcfg.inter_op_parallelism_threads=1
tfcfg.intra_op_parallelism_threads=1
gpuid                       = args.gpuid            # use -1 to choose CPU-only computation.
tf_real_dtype               = tf.float32 if args.revert_to_single_precision else tf.float64
np_real_dtype               = np.float32 if args.revert_to_single_precision else np.float64
tf_complex_dtype            = tf.complex64 if args.revert_to_single_precision else tf.complex128
np_complex_dtype            = np.complex64 if args.revert_to_single_precision else np.complex128

# Architecture
n_layers_lstm               = args.n_layers_lstm
n_layers_fc                 = args.n_layers_fc
n_layers_fc_partially_obs   = args.n_layers_fc_partially_obs

# Training hyperparameters
nEpochs                     = args.n_epochs
batch_size                  = args.batch_size
dropout_keep_prob           = 1.0
batch_norm_MA_decay         = 0.95      # a fraction for decaying the moving average. Reasonable values for decay are close to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc
momentum                    = 0.9       # a constant momentum for SGD
use_nesterov                = True      # If MomentumOptimizer is used, then it will be a nesterov momentum (

# Dataset
pop = create_partial_observability_problem(dataset_dir, dataset_name, args.T, args.Ns, args.Nv, verbose=True, reverse_bus_hiding_order=args.reverse_bus_hiding_order)

nValid                      = int(pop.num_examples_to_generate * (1-pop.test_fraction) * pop.validation_fraction) # number of validation examples.
nTest                       = int(pop.num_examples_to_generate * pop.test_fraction) # number of test examples
nTrain                      = pop.num_examples_to_generate - nValid - nTest
testBatchSize               = batch_size #currently degenerated, but could also be different than the training-batch_size...

# Set seeds (except for the tf seed, which will be asserted in the session)
np.random.seed(random_seed)
random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)

if args.use_complex_tf_ops:
    ld('Error - cannot run complex tensorflow ops. These ops were found unreliable in face of bacpropagation of the gradients.')
    sys.exit(-1)

# Create necessary directories
for dir_t in [log_outputdir, figures_outputdir, tensorboardDir]:
    if not os.path.exists(dir_t):
        os.makedirs(dir_t)

if (not args.no_training) and args.model_type in ['persistent', 'wls']:
    ld("Info: disabling training for a non trainable model type \"{}\".".format(args.model_type))
    args.no_training = True

if args.data_transform != 'none' and args.model_type in ['wls']:
    ld("Info: reverting to non-normalized dataset for a model of a type \"{}\".".format(args.model_type))
    args.data_transform = 'none'

#####################################################
########## Dataset Preparation ######################
#####################################################
# Construct the X1,X2,X2_hidden,y arrays - all in rectangular (cartesian) representation.
X1_train, X2_train, X2hidden_train, y_train, _,X1_test, X2_test, X2hidden_test, y_test, Y_real_np, Y_imag_np, time_steps_test = load_dataset_pop(pop,
                                                                     dtype = np_real_dtype, null_dataset = args.null_dataset)

X1_train, X1_validation, X2_train, X2_validation, X2hidden_train, X2hidden_validation, y_train, y_validation = train_test_split(X1_train, X2_train,
                                                                     X2hidden_train, y_train,
                                                                     test_size=pop.validation_fraction,
                                                                     shuffle=True,
                                                                     random_state=random_seed)

# Normalize / standardize
any_node_hidden  = pop.num_nodes - args.Ns - args.Nv > 0
any_node_visible = args.Ns + args.Nv > 0

if args.data_transform == 'normalize':
    X1_transformer, y_transformer = [NDMinMaxScaler(feature_range=(-1, 1), copy=True), MinMaxScaler((-1, 1))]
    X2_transformer = MinMaxScaler((-1, 1)) if any_node_visible else IdentityScaler()
    X2hidden_transformer =  MinMaxScaler((-1, 1)) if any_node_hidden else IdentityScaler()
elif args.data_transform == 'standardize':
    X1_transformer, y_transformer = [NDStandardScaler(), StandardScaler()]
    X2_transformer = StandardScaler() if any_node_visible else IdentityScaler()
    X2hidden_transformer =  StandardScaler() if any_node_hidden else IdentityScaler()
elif args.data_transform == 'none':
    X1_transformer, X2_transformer, X2hidden_transformer, y_transformer = [IdentityScaler(), IdentityScaler(), IdentityScaler(), IdentityScaler()]

# Normalize/standardize the dataset according to the training data
X1_train = X1_transformer.fit_transform(X1_train)
X1_validation = X1_transformer.transform(X1_validation) if nValid > 0 else X1_validation
X1_test = X1_transformer.transform(X1_test)

X2_train = X2_transformer.fit_transform(X2_train)
X2_validation = X2_transformer.transform(X2_validation) if nValid > 0 else X2_validation
X2_test = X2_transformer.transform(X2_test)

X2hidden_train = X2hidden_transformer.fit_transform(X2hidden_train)
X2hidden_validation = X2hidden_transformer.transform(X2hidden_validation) if nValid > 0 else X2hidden_validation
X2hidden_test = X2hidden_transformer.transform(X2hidden_test)

y_train = y_transformer.fit_transform(y_train)
y_validation = y_transformer.transform(y_validation) if nValid > 0 else y_validation
y_test = y_transformer.transform(y_test)

assert(pop.num_nodes == y_train.shape[1] // 2)
assert( pop.num_target_measurements == y_train.shape[1] )
TensorboardEnabled = TensorboardEnabled_valid or TensorboardEnabled_training
assert(not (TensorboardEnabled_valid and TensorboardEnabled_training))

#####################################################
########## Neural Net Construction ##################
#####################################################
graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(random_seed)
    if gpuid >= 0:
        deviceStr = '/gpu:{gpuid}'.format(gpuid=gpuid)
    else:
        deviceStr = '/cpu:0'
    with tf.device(deviceStr):

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        input_data_X1 = tf.placeholder(tf_real_dtype, shape=(None, args.T-1, pop.num_all_measurements))
        input_data_X2 = tf.placeholder(tf_real_dtype, shape=(None, pop.num_remaining_measurements))
        input_data_X2hidden = tf.placeholder(tf_real_dtype, shape=(None, pop.num_hidden_measurements-pop.num_target_measurements))
        input_labels  = tf.placeholder(tf_real_dtype, shape=(None, pop.num_target_measurements))
        dropout_prob = tf.placeholder(tf_real_dtype) # keep probability. a real value from an interval [0.0,1.0]

        # The topology of the underlying electrical grid is constant and is known in
        # advance in a form of the complex admittance matrix Y = Yrela + 1j * Yimag
        Y_real = tf.constant(Y_real_np, dtype=tf_real_dtype)
        Y_imag = tf.constant(Y_imag_np, dtype=tf_real_dtype)

        # Inverse transform the voltages and the powers.
        denormalized_input_data_X1 = tf_inverse_transform(X1_transformer, input_data_X1, tf_real_dtype)
        denormalized_input_data_X2 = tf_inverse_transform(X2_transformer, input_data_X2, tf_real_dtype)
        denormalized_input_data_X2hidden = tf_inverse_transform(X2hidden_transformer, input_data_X2hidden, tf_real_dtype)
        denormalized_input_labels = tf_inverse_transform(y_transformer, input_labels, tf_real_dtype)

        # Model setting according to command line choice
        if args.model_type == 'neuralnet':
            # Create the neural network computational graph
            lstm_output_dim = math.ceil(pop.num_nodes/3)
            fc_partially_obs_output_dim = math.ceil(pop.num_nodes/6)

            lstm_outputs, final_state = lstm_model(input_data_X1, pop.num_all_measurements, lstm_output_dim,
                                                   n_layers_lstm, dtype=tf_real_dtype, dropout_keep_prob=dropout_prob)
            auxiliary_inputs = fc_model(input_data_X2, pop.num_remaining_measurements, fc_partially_obs_output_dim,
                                        2, dtype=tf_real_dtype, slim=args.slim_auxilliary_network)  #
            intermediate_features = tf.concat([lstm_outputs[:, -1], auxiliary_inputs],
                                              axis=1)
            unscaled_logits = fc_model(intermediate_features, lstm_output_dim + fc_partially_obs_output_dim,
                                       pop.num_target_measurements, n_layers_fc, dtype=tf_real_dtype)
            multiplicative_scaler_vector = fc_model(intermediate_features,
                                                    lstm_output_dim + fc_partially_obs_output_dim,
                                                    pop.num_target_measurements, 1, activation=tf.nn.leaky_relu,
                                                    dtype=tf_real_dtype)
            additive_scaler_vector = fc_model(intermediate_features, lstm_output_dim + fc_partially_obs_output_dim,
                                              pop.num_target_measurements, 1, activation=tf.nn.leaky_relu,
                                              dtype=tf_real_dtype)
            if args.no_trained_scaling:
                logits = unscaled_logits
            else:
                logits = tf.math.add(tf.math.multiply(unscaled_logits, multiplicative_scaler_vector),additive_scaler_vector)

            # Define regularization on all the weights, but not the biases
            L2_norms_of_all_weights_list = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
            weight_norm = tf.add_n(L2_norms_of_all_weights_list)

        elif args.model_type == 'persistent':
            logits = input_data_X1[:, -1, pop.num_nodes * 2:]
            weight_norm = tf.constant(0.0, dtype=tf_real_dtype)

        elif args.model_type == 'wls':
            # Note: the WLS works in a non-normalized environment. Thus we
            # denormalize all the data involved in the computation. The reason:
            # WLS works directly on the Power-Flow equations.


            # The initial voltage guess is the most recently observed voltages.
            # (this is the 'persistent' estimator final guess)
            # TODO for more flexible scenarios, where voltages are also partially observable, the following line should be modified.
            V0_for_WLS = denormalized_input_data_X1[:, -1, pop.num_nodes * 2:] # This is what was submitted to PSCC

            # The powers passed to the WLS solver are partially observable and partially guessed from the previous time step.
            # Carefully concatenate powers appearing in X2 and X1(t-1), as the voltages can be involved in them as well
                                                                                                     # example for hidden power bus list = [0]
            visible_hidden_index_list = pop.visible_power_bus_id_list + pop.hidden_power_bus_id_list # [1,2,3,0]
            visible_hidden_index_array_repeated = np.array(visible_hidden_index_list, dtype=np.int32).repeat(2) * 2 #[2,2,4,4,6,6,0,0]
            visible_hidden_index_array_repeated[1::2] = visible_hidden_index_array_repeated[0::2] + 1  #[2,3,4,5,6,7,0,1]
            inverted_index_mapping_dict = {k: v for v, k in enumerate(visible_hidden_index_list)} # {1:0, 2:1, 3:2, 0:3}
            inverted_visible_hidden_index_list = [inverted_index_mapping_dict[k] for k in range(len(visible_hidden_index_list))] # [3,0,1,2]
            inverted_visible_hidden_index_array = np.array(inverted_visible_hidden_index_list, dtype=np.int32)
            inverted_visible_hidden_index_array_repeated = inverted_visible_hidden_index_array.repeat(2) * 2
            inverted_visible_hidden_index_array_repeated[1::2] = inverted_visible_hidden_index_array_repeated[0::2] + 1 # [6,7,0,1,2,3,4,5]

            power_reordering_indices_WLS = tf.constant(value=inverted_visible_hidden_index_array_repeated,
                                                       dtype=tf.int32)
            guessed_powers_from_X1 = tf.gather(params=denormalized_input_data_X1[:, -1, :],
                                               indices=visible_hidden_index_array_repeated[2 * len(pop.visible_power_bus_id_list):], axis=1)#

            known_powers_from_X2 = denormalized_input_data_X2

            unordered_powers_WLS = tf.concat([known_powers_from_X2, guessed_powers_from_X1], axis=1)
            S_for_WLS = tf.gather(unordered_powers_WLS, power_reordering_indices_WLS, axis=1)

            # Compute a weight for each powerflow equation
            tf_epsilon = tf.constant(value=0.000000000001, dtype=tf_real_dtype)
            _, stds_for_weights = tf.nn.moments(denormalized_input_data_X1[:,:,:2*pop.num_nodes], axes=[1]) # (B x 72)
            unprocessed_stds_for_WLS = stds_for_weights[:, 0::2] + stds_for_weights[:, 1::2]+tf_epsilon   # (B x 36)
            min_std_per_example = tf.reduce_min(unprocessed_stds_for_WLS, axis=1) # (B)
            min_std_per_example_replicated = tf.reshape(tf.transpose(tf.tile(tf.transpose(min_std_per_example),
                                                                             multiples=[pop.num_nodes])),
                                                        shape=tf.shape(unprocessed_stds_for_WLS)) # (B x 36)
            unordered_mask = tf.constant([1.0 for _ in range(len(pop.visible_power_bus_id_list))] +
                                         [0.0 for _ in range(len(pop.hidden_power_bus_id_list))], dtype=tf_real_dtype) # (36)
            visible_mask = tf.gather(unordered_mask,indices=inverted_visible_hidden_index_array) # (36)
            visible_mask_repeated = tf.reshape(tf.tile(visible_mask, multiples=[tf.shape(denormalized_input_labels)[0]]),
                                               shape=tf.shape(unprocessed_stds_for_WLS)) # (B x 36)
            visible_stds = tf.multiply(min_std_per_example_replicated/10, visible_mask_repeated) # (B x 36) # as an ugly heuristic - we decided to take the minimum of the STDs and decrease it by an additional factor of 10 and this would be the STD of the observable measurements
            hidden_stds = tf.multiply(unprocessed_stds_for_WLS, tf.subtract(tf.constant(1.0, dtype=tf_real_dtype), visible_mask_repeated)) # (B x 36)
            stds_for_WLS = hidden_stds + visible_stds # (B x 36) # combine the observable and the non observable STDs
            weights_for_WLS = tf.reciprocal(stds_for_WLS) # (B x 36) # Weights are reciprocal of the STDs This should be a B*num_nodes shaped tensor
            if args.wls_weights_disabled:
                weights_for_WLS = tf.ones_like(weights_for_WLS, dtype=tf_real_dtype)
            elif args.wls_weights_discriminate_hidden:
                weights_for_WLS = tf.divide(weights_for_WLS, tf.reduce_max(weights_for_WLS))   # Example for Ns=35 W=[1, ........ 1, 1/std ]
            else:
                weights_for_WLS = tf.reciprocal(unprocessed_stds_for_WLS)

            # Invoke the tensorflowed version of the scipy's non linear least-squares solver
            #denormalized_logits = V0_for_WLS # ramaut mi seder sheni
            if args.wls_with_power:
                denormalized_logits = tf.py_func(func=solve_wls_with_power,
                                                 inp=[known_powers_from_X2, guessed_powers_from_X1, V0_for_WLS, Y_real, Y_imag, weights_for_WLS, power_reordering_indices_WLS],
                                                 Tout=input_labels.dtype, stateful=False, name="Scipy_wls_solver")
            else:
                denormalized_logits = tf.py_func(func=solve_wls,
                                                 inp=[S_for_WLS, V0_for_WLS, Y_real, Y_imag, weights_for_WLS],
                                                 Tout=input_labels.dtype, stateful=False, name="Scipy_wls_solver")
            # return to normalized data state:
            logits = tf_transform(y_transformer, denormalized_logits, tf_real_dtype)

            # Define regularization on all the weights, but not the biases
            weight_norm = tf.constant(0.0, dtype=tf_real_dtype)
        else:
            ld("Error, model type \"{}\" is not supported".format(args.model_type))

        # Power flow equations as a regularizer:
        #  s: complex power vectors (batch_size x n_nodes x 1)
        #  v: complex voltage vectors of the ground truth voltages (batch_size x n_nodes x 1)
        #  v_est: complex voltage vectors of the estimated voltages (batch_size x n_nodes x 1)
        #  Y: complex admittance matrix (n_nodes x n_nodes)
        real_indices = np.array(range(0, pop.num_target_measurements - 1,2))
        imag_indices = np.array(range(1, pop.num_target_measurements, 2))

        # Carefully concatenate powers appearing in X2 and X2hidden, as the voltages can be involved in them as well
        visible_hidden_index_list = pop.visible_power_bus_id_list + pop.hidden_power_bus_id_list
        inverted_index_mapping_dict = {k:v for v,k in enumerate(visible_hidden_index_list)}
        inverted_visible_hidden_index_list = [inverted_index_mapping_dict[k] for k in range(len(visible_hidden_index_list))]
        array_mapping_X2_and_X2hidden_to_unified_power_array = np.array(inverted_visible_hidden_index_list, dtype=np.int32)
        array_mapping_X2_and_X2hidden_to_unified_power_array_repeated = array_mapping_X2_and_X2hidden_to_unified_power_array.repeat(2)*2
        array_mapping_X2_and_X2hidden_to_unified_power_array_repeated[1::2] = array_mapping_X2_and_X2hidden_to_unified_power_array_repeated[0::2]+1
        power_reordering_indices = tf.constant(value=array_mapping_X2_and_X2hidden_to_unified_power_array_repeated, dtype=tf.int32)
        unordered_powers = tf.concat([denormalized_input_data_X2[:,:pop.num_visible_power_measurements], denormalized_input_data_X2hidden[:,:pop.num_hidden_power_measurements]], axis=1)
        transposed_unordered_powers = tf.transpose(unordered_powers) # 2*num_nodes x batch_size
        transposed_s_real_imag_interleaved = tf.gather(transposed_unordered_powers, power_reordering_indices)
        s_real_imag_interleaved = tf.transpose(transposed_s_real_imag_interleaved)
        s_real = tf.gather(s_real_imag_interleaved, real_indices, axis=1)
        s_imag = tf.gather(s_real_imag_interleaved, imag_indices, axis=1)
        v_real = tf.gather(denormalized_input_labels, real_indices, axis=1)
        v_imag = tf.gather(denormalized_input_labels, imag_indices, axis=1)
        denormalized_logits = tf_inverse_transform(y_transformer, logits, tf_real_dtype)
        v_est_real = tf.gather(denormalized_logits, real_indices, axis=1)
        v_est_imag = tf.gather(denormalized_logits, imag_indices, axis=1)

        if args.use_complex_tf_ops:
            s = tf.complex(s_real, s_imag)
            v = tf.complex(v_real, v_imag)
            v_est = tf.complex(v_est_real, v_est_imag)
            Y = tf.complex(Y_real, Y_imag)
            pfe_numeric_mse, pfe_numeric_mae = power_flow_equations_loss_complex(s, v, Y, args.batch_size, tf_real_dtype)
            pfe_loss, _ = power_flow_equations_loss_complex(s, v_est, Y, args.batch_size, tf_real_dtype)
        else:
            pfe_numeric_mse, pfe_numeric_mae = power_flow_equations_loss_real(s_real, s_imag, v_real, v_imag, Y_real, Y_imag, args.batch_size, tf_real_dtype)
            pfe_loss, _ = power_flow_equations_loss_real(s_real, s_imag, v_est_real, v_est_imag, Y_real, Y_imag, args.batch_size, tf_real_dtype)
        per_node_denormalized_mse = tf.reduce_mean(tf.square(tf.subtract(denormalized_input_labels, denormalized_logits)),axis=0)
        per_node_denormalized_mae = tf.reduce_mean(tf.math.abs(tf.subtract(denormalized_input_labels, denormalized_logits)),axis=0)


        regularizing_term = args.weight_decay * weight_norm + \
                            args.equations_regularizer * pfe_loss

        per_node_mse = tf.reduce_mean(tf.square(tf.subtract(input_labels, logits)),axis=0)
        per_node_mae = tf.reduce_mean(tf.math.abs(tf.subtract(input_labels, logits)),axis=0)

        mse = tf.reduce_mean(per_node_mse)
        mae = tf.reduce_mean(per_node_mae)
        loss = mse + regularizing_term
        predictions = logits

        if args.model_type == 'neuralnet':
            # Optimizer creation
            optimizer = tf.train.AdamOptimizer()

            # Gradients: computation, clipping (to avoid exploding gradients) and application thereof (to update the weights)
            grads_unclipped = optimizer.compute_gradients(loss, tf.trainable_variables())
            grads = [(tf.clip_by_value(gv[0], -1, 1), gv[1]) for gv in grads_unclipped]
            optimizer = optimizer.apply_gradients(grads)
        else:
            optimizer = tf.constant(0.0, dtype=tf_real_dtype)

        # An initializer op is defined here. It is later to be invoked, prior
        # to training.
        init_op = tf.global_variables_initializer()

    if TensorboardEnabled_training:
        with tf.name_scope('train_curves'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('MSE', mse)
            tf.summary.scalar('MAE', mae)
            tf.summary.scalar('PFE_loss', pfe_loss)
            tf.summary.scalar('PFE_numeric_MSE', pfe_numeric_mse)
            tf.summary.scalar('PFE_numeric_MAE', pfe_numeric_mae)

        with tf.name_scope('Nodewise_MSE'):
            for nnode in range(pop.num_nodes):
                tf.summary.scalar('ReV{}'.format(nnode+1), per_node_mse[2*nnode])
                tf.summary.scalar('ImV{}'.format(nnode + 1), per_node_mse[2 * nnode + 1])
        with tf.name_scope('Nodewise_MAE'):
            for nnode in range(pop.num_nodes):
                tf.summary.scalar('ReV{}'.format(nnode+1), per_node_mae[2*nnode])
                tf.summary.scalar('ImV{}'.format(nnode + 1), per_node_mae[2 * nnode + 1])
        with tf.name_scope('Nodewise_Denormalized_MSE'):
            for nnode in range(pop.num_nodes):
                tf.summary.scalar('ReV{}'.format(nnode+1), per_node_denormalized_mse[2*nnode])
                tf.summary.scalar('ImV{}'.format(nnode + 1), per_node_denormalized_mse[2 * nnode + 1])
        with tf.name_scope('Nodewise_Denormalized_MAE'):
            for nnode in range(pop.num_nodes):
                tf.summary.scalar('ReV{}'.format(nnode+1), per_node_denormalized_mae[2*nnode])
                tf.summary.scalar('ImV{}'.format(nnode + 1), per_node_denormalized_mae[2 * nnode + 1])

        if args.model_type == 'neuralnet':
            with tf.name_scope('gradient_summary'):
                for grad, var in grads:
                    if grad is not None:
                        gradName = var.name.replace(":", "_")
                        with tf.name_scope(gradName):
                            tf.summary.scalar('sparsity', tf.nn.zero_fraction(grad))
                            mean = tf.reduce_mean(grad)
                            tf.summary.scalar('mean', mean)
                            with tf.name_scope('stddev'):
                                stddev = tf.sqrt(tf.reduce_mean(tf.square(grad - mean)))
                            tf.summary.scalar('stddev', stddev)
                            tf.summary.scalar('max', tf.reduce_max(grad))
                            tf.summary.scalar('min', tf.reduce_min(grad))
                            tf.summary.histogram('histogram', grad)

            with tf.name_scope('weights_summary'):
                for var_i in tf.trainable_variables():
                    varName = var_i.name
                    try:
                        with tf.name_scope(varName):
                            mean = tf.reduce_mean(var_i)
                            tf.summary.scalar('mean', mean)
                            with tf.name_scope('stddev'):
                                stddev = tf.sqrt(tf.reduce_mean(tf.square(var_i - mean)))
                            tf.summary.scalar('stddev', stddev)
                            tf.summary.scalar('max', tf.reduce_max(var_i))
                            tf.summary.scalar('min', tf.reduce_min(var_i))
                            tf.summary.histogram('histogram', var_i)
                    except:
                        pass

    if TensorboardEnabled_valid:
        with tf.name_scope('test_curves'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('MSE', mse)

    if TensorboardEnabled:
        summary = tf.summary.merge_all()

if TensorboardEnabled:
    summary_writer = tf.summary.FileWriter(os.path.join(tensorboardDir,model_name), graph=graph)

if not args.no_training:
    with tf.Session(graph=graph, config=tfcfg) as session:

        if args.revert_to_single_precision:
            ld("Using SINGLE precision.")
        else:
            ld("Using DOUBLE precision.")

        if args.model_type == 'neuralnet':
            saver = tf.train.Saver()
            if restore_session:
                actionTaken = "Loaded existing "
                saver.restore(session, "Models/" + model_name_for_loading + ".ckpt")
            else:
                actionTaken = "Initialized new "
                session.run(init_op)
            nParams = countParams()
            ld(actionTaken + "model, consisting of " + str(nParams) + " parameters. Beginning training...")
        else:
            ld("A non-trainable model type \"{}\" will not be loaded/dumped to a file.".format(args.model_type))

        summary_writer_epoch_counter = 0

        nBatchesPerEpoch = int(ceil(X1_train.shape[0] / float(batch_size)))

        ld("Training {} epochs. TrainSet contains {} examples.".format(nEpochs,X1_train.shape[0]))

        # Prepare statistics
        trainMSE = np.zeros(nEpochs)
        validMSE = np.zeros(nEpochs)
        validPFEE = np.zeros(nEpochs)
        validPFENMSE = np.zeros(nEpochs)
        validPFENMAE = np.zeros(nEpochs)
        trainLosses = np.zeros(nEpochs)
        validLosses = np.zeros(nEpochs)
        trainPerNodeMses = np.zeros((nEpochs, pop.num_target_measurements))
        validPerNodeMses = np.zeros((nEpochs, pop.num_target_measurements))

        # Open Log File
        logFile = open(os.path.join(log_outputdir, model_name+".txt"), "w")
        MSE_File = open(os.path.join(log_outputdir, model_name + "_test_MSE.txt"), "w")
        printLogHeader(logFile, nValid > 0)
        printPernodeMSEHeader(MSE_File, ["ReV{}".format(nodid//2+1) if nodid % 2 == 0 else "ImV{}".format(nodid//2+1) for nodid in range(0,2*pop.num_nodes)])#["ReV1","ImV1","ReV2","ImV2","ReV3","ImV3","ReV4","ImV4"])
        logLine = 1

        minLoss = 999999.9
        num_progressbar_steps = X1_train.shape[0] * nEpochs
        bar = progressbar.ProgressBar(maxval=num_progressbar_steps, widgets=[progressbar.Bar('=', '[', ']'),
                                                                             ' ', progressbar.Percentage(),
                                                                             ' ', progressbar.ETA()])
        bar.start()
        bar_counter = 0

        # Grand Training <--> Evaluating loop
        for epoch in range(nEpochs):

            if args.model_type != 'neuralnet':
                if epoch == 0:
                    ld("Skipping all the training epochs for model type {}".format(args.model_type))
                continue

            # Shuffle the <original> training data
            X1_train, X2_train, X2hidden_train, y_train = shuffle(X1_train, X2_train, X2hidden_train, y_train, random_state=random_seed)

            # Do optimization iterations of the current epoch
            for step in range(nBatchesPerEpoch):

                # Pick an offset within the training data, which has been randomized.
                # Generate a minibatch and prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it...
                sampleId_low = step*batch_size
                sampleId_high= min(sampleId_low + batch_size, X1_train.shape[0]) # the minimum is taken to prevent the overflowing of indices in the last batch (which is smaller than others)
                feed_dict = {input_data_X1: X1_train[sampleId_low:sampleId_high],
                             input_data_X2: X2_train[sampleId_low:sampleId_high],
                             input_data_X2hidden: X2hidden_train[sampleId_low:sampleId_high],
                             input_labels: y_train[sampleId_low:sampleId_high],
                             dropout_prob: dropout_keep_prob}

                if TensorboardEnabled_training:
                    summary_str, _, batch_loss, batch_mse, batch_pernode_loss= session.run([summary, optimizer, loss, mse, per_node_mse],
                                                                        feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, summary_writer_epoch_counter)
                    summary_writer.flush()
                    summary_writer_epoch_counter += 1
                else:
                    _, batch_loss, batch_mse, batch_pernode_loss = session.run([optimizer, loss, mse, per_node_mse],
                                                feed_dict=feed_dict)

                # Maintain statistics and progress-bar
                trainLosses[epoch]   += batch_loss
                trainMSE[epoch] += batch_mse
                trainPerNodeMses[epoch] += batch_pernode_loss
                bar_counter += (sampleId_high - sampleId_low)
                bar.update(bar_counter)

            # Epoch ended, validate model
            val_loss = AverageTracker(0.0)
            val_mse = AverageTracker(0.0)
            val_pfee = AverageTracker(0.0)
            val_pfenmse = AverageTracker(0.0)
            val_pfenmae = AverageTracker(0.0)
            val_PerNodeMseVector = AverageTracker(np.zeros_like(y_test[0]))
            nTestBatches = int(ceil(nTest / float(testBatchSize)))
            for tb in range(nTestBatches):
                sample_low = tb * testBatchSize
                sample_high = min(sample_low + testBatchSize, X1_test.shape[0])
                if nValid > 0:
                    feed_dict = {input_data_X1: X1_validation[sample_low:sample_high],
                                 input_data_X2: X2_validation[sample_low:sample_high],
                                 input_data_X2hidden: X2hidden_validation[sample_low:sample_high],
                                 input_labels: y_validation[sample_low:sample_high],
                                 dropout_prob: 1.0}
                else:
                    feed_dict = {input_data_X1: X1_test[sample_low:sample_high],
                                 input_data_X2: X2_test[sample_low:sample_high],
                                 input_data_X2hidden: X2hidden_test[sample_low:sample_high],
                                 input_labels: y_test[sample_low:sample_high],
                                 dropout_prob: 1.0}

                if TensorboardEnabled_valid:
                    summary_str, val_loss_t, val_mse_t, per_node_mse_t = session.run([summary, loss, mse, per_node_mse], feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, summary_writer_epoch_counter)
                    summary_writer.flush()
                    summary_writer_epoch_counter += 1
                else:
                    [val_loss_t, val_mse_t, per_node_mse_t, val_pfee_t, val_pfenmse_t, val_pfenmae_t] = session.run([loss, mse, per_node_mse, pfe_loss, pfe_numeric_mse, pfe_numeric_mae], feed_dict=feed_dict)
                val_loss.update(val_loss_t)
                val_mse.update(val_mse_t)
                val_pfee.update(val_pfee_t)
                val_pfenmse.update(val_pfenmse_t)
                val_pfenmae.update(val_pfenmae_t)
                val_PerNodeMseVector.update(per_node_mse_t)

            validLosses[epoch] = val_loss.mean()
            validMSE[epoch] = val_mse.mean()
            validPFEE[epoch] = val_pfee.mean()
            validPFENMSE[epoch] = val_pfenmse.mean()
            validPFENMAE[epoch] = val_pfenmae.mean()
            validPerNodeMses[epoch] = val_PerNodeMseVector.mean()


            # Checkpoint the model each time the validation accuracy is improved
            if args.model_type == 'neuralnet' and minLoss < validLosses[epoch]:
                saver.save(session, "Models/" + model_name + "_best.ckpt")
                minLoss = validLosses[epoch]
            val_test_str = " validation " if nValid > 0 else " test"
            ld("Epoch " + str(epoch) + val_test_str +
               " loss {:3.6f} mse {:.3e} pfee {:.3e} pfenmse {:.3e} pfenmae {:.3e}"
               .format(validLosses[epoch], validMSE[epoch], validPFEE[epoch], validPFENMSE[epoch], validPFENMAE[epoch]))
            for nnode in range(pop.num_nodes):
                ld(" - Test MSE of Re(V{}): {:.8f}".format(nnode + 1, validPerNodeMses[epoch, 2 * nnode]))
                ld(" - Test MSE of Im(V{}): {:.8f}".format(nnode + 1, validPerNodeMses[epoch, 2 * nnode + 1]))

            # Maintain log file after each epoch
            # Must normalize the training measurements, since they were summed over all the batches
            trainLosses[epoch]   /= nBatchesPerEpoch
            printLogEntry(logFile, logLine, validLosses[epoch], trainLosses[epoch], validMSE[epoch], trainMSE[epoch])
            printPernodeEntry(MSE_File, logLine, validPerNodeMses[epoch])
            logLine += 1

            # In case when it's a full-data (no validation) run - save the model every 10 epochs
            if args.model_type == 'neuralnet':
                if epoch % 10 == 9 or epoch == (nEpochs - 1):
                    saver.save(session, "Models/"+model_name+".ckpt")

        # Training complete
        bar.finish()


        # Calculate Test Accuracy for the BEST MODEL (requires restoration of the best saved session)
        if nValid > 0 and args.model_type == 'neuralnet':
            if minLoss < validLosses[nEpochs - 1]:
                ld("Restoring the best model based on validation loss (epoch " + str((np.argmax(validLosses))) + ")")
                saver.restore(session, "Models/" + model_name + "_best.ckpt")
            else:
                ld("Using the latest epoch's models since it showed the lowest validation loss")
        else:
            ld("No validation performed during this training. Calculating test accuracy based on the last epoch.")

        nTestBatches = int(ceil(nTest / float(testBatchSize)))
        testLoss, testMSE, testMAE = AverageTracker(0.0), AverageTracker(0.0), AverageTracker(0.0)
        testPerNodeMseVector, testPerNodeMaeVector = AverageTracker(np.zeros_like(y_test[0])), AverageTracker(np.zeros_like(y_test[0]))
        denormalized_testMSE, denormalized_testMAE = AverageTracker(0.0), AverageTracker(0.0)
        denormalized_testPerNodeMseVector, denormalized_testPerNodeMaeVector = AverageTracker(np.zeros_like(y_test[0])), AverageTracker(np.zeros_like(y_test[0]))

        bar = progressbar.ProgressBar(maxval=nTestBatches, widgets=[progressbar.Bar('=', '[', ']'),
                                                                    ' ', progressbar.Percentage(),
                                                                    ' ', progressbar.ETA()])
        bar.start()
        for tb in range(nTestBatches):
            bar.update(tb)
            sample_low = tb * testBatchSize
            sample_high = min(sample_low + testBatchSize, X1_test.shape[0])
            feed_dict = {input_data_X1: X1_test[sample_low:sample_high],
                         input_data_X2: X2_test[sample_low:sample_high],
                         input_data_X2hidden: X2hidden_test[sample_low:sample_high],
                         input_labels: y_test[sample_low:sample_high],
                         dropout_prob:1.0}
            [testLoss_t, testMSE_t, testMAE_t, per_node_mse_t, per_node_mae_t, predictions_t] = session.run([loss, mse, mae, per_node_mse, per_node_mae, predictions], feed_dict=feed_dict)
            testLoss.update(testLoss_t)
            testMSE.update(testMSE_t)
            testMAE.update(testMAE_t)
            testPerNodeMseVector.update(per_node_mse_t)
            testPerNodeMaeVector.update(per_node_mae_t)

            # Denormalize the data
            denormalized_labels = y_transformer.inverse_transform(y_test[sample_low:sample_high])
            denormalized_predictions = y_transformer.inverse_transform(predictions_t)
            real_indices = np.array(range(0, pop.num_target_measurements - 1, 2))
            imag_indices = np.array(range(1, pop.num_target_measurements, 2))
            label_real = np.take(denormalized_labels, real_indices, axis=1)
            prediction_real = np.take(denormalized_predictions, real_indices, axis=1)
            label_imag = np.take(denormalized_labels, imag_indices, axis=1)
            prediction_imag = np.take(denormalized_predictions, imag_indices, axis=1)
            mag_prediction, ang_prediction = realImagVectorsToMagAngVectors(prediction_real, prediction_imag)
            mag_label, ang_label = realImagVectorsToMagAngVectors(label_real, label_imag)
            ground_truths = np.concatenate((mag_label, ang_label),1)
            predicteds = np.concatenate((mag_prediction, ang_prediction),1)
            denormalized_NumpyPerNodeMseVector = mse_func(ground_truths, predicteds)
            denormalized_NumpyPerNodeMaeVector = mae_func(ground_truths, predicteds)
            denormalized_testMSE.update(np.mean(denormalized_NumpyPerNodeMseVector))
            denormalized_testPerNodeMseVector.update(denormalized_NumpyPerNodeMseVector)
            denormalized_testMAE.update(np.mean(denormalized_NumpyPerNodeMaeVector))
            denormalized_testPerNodeMaeVector.update(denormalized_NumpyPerNodeMaeVector)

        bar.finish()
        # Create plots of (train, validation) if validation exists, else, (train, test)
        validStr = 'Validation' if nValid > 0 else 'Test'
        plotTwoLines(trainLosses, validLosses, range(1,nEpochs+1),
                    "Epoch", "Loss", validStr + " Loss", figures_outputdir, model_name+"_Loss.jpg",
                    isAnnotatedMin=False, validStr=validStr)
        plotTwoLines(trainMSE, validMSE, range(1,nEpochs+1),
                    "Epoch", "MSE(%)", "Mean Squared Error (Test=%.8f" % testMSE.mean() + ")", figures_outputdir, model_name + "_MSE.jpg",
                    isAnnotatedMin=True, validStr=validStr)

        ld("Test (average over all nodes) MSE: {:.3e} MAE: {:.3e}".format(testMSE.mean(),testMAE.mean()))
        for nnode in range(pop.num_target_measurements//pop.num_measurements_per_phasor):
            ld(" - For Re(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, testPerNodeMseVector.mean()[2*nnode], testPerNodeMaeVector.mean()[2*nnode]))
            ld(" - For Im(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, testPerNodeMseVector.mean()[2*nnode+1], testPerNodeMaeVector.mean()[2*nnode+1]))
        ld("Denormalized Test (average over all nodes) MSE: {:.3e} MAE: {:.3e}".format(denormalized_testMSE.mean(),denormalized_testMAE.mean()))
        for nnode in range(pop.num_target_measurements//pop.num_measurements_per_phasor):
            ld(" - For Mag(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, denormalized_testPerNodeMseVector.mean()[2*nnode], denormalized_testPerNodeMaeVector.mean()[2*nnode]))
            ld(" - For Ang(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, denormalized_testPerNodeMseVector.mean()[2*nnode+1], denormalized_testPerNodeMaeVector.mean()[2*nnode+1]))


        testLossStr = "Test MSE: normalized: {:.20f} denormalized {:.20f}".format(testMSE.mean(),denormalized_testMSE.mean())
        ld(testLossStr)
        logFile.write(testLossStr)
        logFile.close()
        MSE_File.close()
        ld("************************************************************")

        # Final model saving (after all the trainings)
        if args.model_type == 'neuralnet':
            saver.save(session, "Models/"+model_name+".ckpt")



ld("Model Final Testing")
MSE_per_example_File = open(os.path.join(log_outputdir, model_name + "_test_MSE_per_example.txt"), "w")
MAE_per_example_File = open(os.path.join(log_outputdir, model_name + "_test_MAE_per_example.txt"), "w")
log_column_headers_list = ["MagV{} ".format(nodid // 2 + 1) if nodid % 2 == 0 else "AngV{} ".format(nodid // 2 + 1) for nodid in range(0, 2 * pop.num_nodes)]
printPernodeMSEHeader(MSE_per_example_File, log_column_headers_list)
printPernodeMSEHeader(MAE_per_example_File, log_column_headers_list)
logLine = 1

with tf.Session(graph=graph, config=tfcfg) as session:
    # Restore variables from disk.

    if args.model_type == 'neuralnet':
        saver = tf.train.Saver()
        saver.restore(session, "Models/" + model_name + ".ckpt")
        nParams = countParams()
        ld("Model restored from Models/" + model_name + ".ckpt, having " + str(nParams) + " parameters")

    nTestBatches = int(ceil(nTest / float(testBatchSize)))
    testLoss, testMSE, testMAE = AverageTracker(0.0), AverageTracker(0.0), AverageTracker(0.0)
    testPerNodeMseVector, testPerNodeMaeVector = AverageTracker(np.zeros_like(y_test[0])), AverageTracker(np.zeros_like(y_test[0]))
    denormalized_testMSE, denormalized_testMAE = AverageTracker(0.0), AverageTracker(0.0)
    denormalized_testPerNodeMseVector, denormalized_testPerNodeMaeVector = AverageTracker(np.zeros_like(y_test[0])), AverageTracker(np.zeros_like(y_test[0]))
    chronological_denormalized_ground_truths = np.zeros((time_steps_test.shape[0], pop.num_target_measurements), dtype=np_real_dtype)
    chronological_denormalized_predictions = np.zeros((time_steps_test.shape[0], pop.num_target_measurements), dtype=np_real_dtype)

    if not args.no_prediction_plots:
        predict_image_dir = os.path.join("Figures", "Predictions_" + model_name)
        if not os.path.exists(predict_image_dir):
            os.makedirs(predict_image_dir)

    # indices indicating the locations of the various measurements
    # inside the input (x1) vector and the predictions vector
    # These indices will be used for satterring and gathering of the results.
    V_real_indices_in_X1 = list(range(2 * pop.num_nodes, 4 * pop.num_nodes, 2))
    V_imag_indices_in_X1 = list(range(1 + 2 * pop.num_nodes, 4 * pop.num_nodes, 2))
    V_real_indices_in_labels_or_predictions = list(range(0, pop.num_target_measurements, 2))
    V_imag_indices_in_labels_or_predictions = list(range(1, pop.num_target_measurements, 2))

    bar = progressbar.ProgressBar(maxval=nTestBatches, widgets=[progressbar.Bar('=', '[', ']'),
                                                                         ' ', progressbar.Percentage(),
                                                                         ' ', progressbar.ETA()])
    bar.start()
    for tb in range(nTestBatches):
        bar.update(tb)
        sample_low = tb * testBatchSize
        sample_high = min(sample_low + testBatchSize, X1_test.shape[0])
        feed_dict = {input_data_X1: X1_test[sample_low:sample_high],
                     input_data_X2: X2_test[sample_low:sample_high],
                     input_data_X2hidden: X2hidden_test[sample_low:sample_high],
                     input_labels: y_test[sample_low:sample_high],
                     dropout_prob:1.0}
        [testLoss_t, testMSE_t, testMAE_t, predictions_t, per_node_mse_t, per_node_mae_t] = session.run([loss, mse, mae, predictions, per_node_mse, per_node_mae], feed_dict=feed_dict)
        testLoss.update(testLoss_t)
        testMSE.update(testMSE_t)
        testPerNodeMseVector.update(per_node_mse_t)
        testMAE.update(testMAE_t)
        testPerNodeMaeVector.update(per_node_mae_t)

        # INVERSE_TRANSFORM of the datasets! Both the predicted and the ground truth.
        X1_test[sample_low:sample_low+predictions_t.shape[0]] = X1_transformer.inverse_transform(X1_test[sample_low:sample_low+predictions_t.shape[0]])
        predictions_t = y_transformer.inverse_transform(predictions_t)
        y_test[sample_low:sample_low+predictions_t.shape[0]] = y_transformer.inverse_transform(y_test[sample_low:sample_low+predictions_t.shape[0]])
        for i in range(predictions_t.shape[0]):

            global_sample_id = sample_low + i

            # TODO: Slack bus measurements are to be excluded from the presentation.
            v_mag = np.zeros((pop.num_target_buses, args.T), dtype=np_real_dtype)
            v_ang = np.zeros((pop.num_target_buses, args.T), dtype=np_real_dtype)
            v_mag_pred = np.zeros(pop.num_target_buses, dtype=np_real_dtype)
            v_ang_pred = np.zeros(pop.num_target_buses, dtype=np_real_dtype)

            index_lists = zip(list(range(pop.num_target_buses)), V_real_indices_in_X1, V_imag_indices_in_X1, V_real_indices_in_labels_or_predictions, V_imag_indices_in_labels_or_predictions)
            for bus_idx,real_x1_idx, imag_x1_idx, real_y_idx, imag_y_idx in index_lists:
                v_mag[bus_idx,:-1]  = X1_test[global_sample_id, :, real_x1_idx]
                v_mag[bus_idx, -1]  = y_test[global_sample_id, real_y_idx]
                v_mag_pred[bus_idx] = predictions_t[i,real_y_idx]
                v_ang[bus_idx,:-1]  = X1_test[global_sample_id, :, imag_x1_idx]
                v_ang[bus_idx, -1]  = y_test[global_sample_id, imag_y_idx]
                v_ang_pred[bus_idx] = predictions_t[i,imag_y_idx]

                # Force magnitude-angle representaton for the voltages
                v_mag[bus_idx], v_ang[bus_idx] = realImagVectorsToMagAngVectors(v_mag[bus_idx], v_ang[bus_idx])
            v_mag_pred, v_ang_pred = realImagVectorsToMagAngVectors(v_mag_pred, v_ang_pred)

            # Gather the Mag,Ang vectors through interleaving
            ground_truths = np.zeros(pop.num_target_measurements, dtype=np_real_dtype)
            predicteds = np.zeros(pop.num_target_measurements, dtype=np_real_dtype)
            np.put(ground_truths, V_real_indices_in_labels_or_predictions, v_mag[:, -1])
            np.put(ground_truths, V_imag_indices_in_labels_or_predictions, v_ang[:, -1])
            np.put(predicteds, V_real_indices_in_labels_or_predictions, v_mag_pred)
            np.put(predicteds, V_imag_indices_in_labels_or_predictions, v_ang_pred)

            chronological_denormalized_ground_truths[global_sample_id, :] = ground_truths
            chronological_denormalized_predictions[global_sample_id, :] = predicteds

            denormalized_NumpyPerNodeMseVector = se_func(ground_truths, predicteds)
            denormalized_NumpyPerNodeMaeVector = ae_func(ground_truths, predicteds)
            denormalized_testMSE.update(np.mean(denormalized_NumpyPerNodeMseVector))
            denormalized_testPerNodeMseVector.update(denormalized_NumpyPerNodeMseVector)
            denormalized_testMAE.update(np.mean(denormalized_NumpyPerNodeMaeVector))
            denormalized_testPerNodeMaeVector.update(denormalized_NumpyPerNodeMaeVector)

            printPernodeEntry(MSE_per_example_File, logLine, denormalized_NumpyPerNodeMseVector)
            printPernodeEntry(MAE_per_example_File, logLine, denormalized_NumpyPerNodeMaeVector)
            logLine += 1


            magword = "Magnitude"
            angleword = "Angle"
            voltage_angle_units = "Degrees"
            max_buses_per_figure = 6
            if not args.no_prediction_plots:
                # Set up neat appearance for the plot.
                extras_dict = {}
                extras_dict["marker_list"] = [MarkerStyle(marker=".", fillstyle='none') if i%2==0 else MarkerStyle(marker="+", fillstyle='full') for i in range(pop.num_target_measurements)]#[".", "+"] * (number_of_nodes_to_estimate)
                extras_dict["marker_scaler_list"] = [40, 20] * (pop.num_nodes)
                extras_dict["legend_location"] = 6
                extras_dict['font_size'] = 14
                legends_lst = []

                for jk in range(pop.num_nodes):
                    legends_lst += [r"$V_{" + str(1+jk) + r"}(0:" + str(args.T - 1) +r")$", r"$\hat V_{" + str(1+jk) + r"}(" + str(args.T - 1) + r")$"]

                # Break the figures into multiple ones
                for plot_set in range(int(ceil(pop.num_target_buses / max_buses_per_figure))):
                    bus_id_low = plot_set * max_buses_per_figure
                    bus_id_high= min(pop.num_target_buses, (plot_set+1) * max_buses_per_figure)
                    plot_id_low = 2*bus_id_low
                    plot_id_high= 2*bus_id_high
                    num_plots_in_this_figure = plot_id_high-plot_id_low
                    extra_string_for_filename = "" if bus_id_low==0 and bus_id_high==pop.num_target_buses else "buses_{}-{}_".format(bus_id_low+1,bus_id_high)
                    plotListOfScatters(([list(range(args.T))] + [[args.T-1]]) * num_plots_in_this_figure,
                                       #[v1_mag, v1_mag_pred, v2_mag, v2_mag_pred,v3_mag, v3_mag_pred, v4_mag, v4_mag_pred],
                                       [v_mag[bus_idx//2] if bus_idx%2==0 else v_mag_pred[bus_idx//2] for bus_idx in range(plot_id_low,plot_id_high)],
                                       legends_lst[plot_id_low:plot_id_high],
                                       "t (minutes)", "Volts",
                                       "Test Example {} - {}".format(global_sample_id, magword),
                                       predict_image_dir,
                                       "Predict_{:03d}_{}{}V.png".format(global_sample_id,extra_string_for_filename, magword),
                                       extras_dict)
                    plotListOfScatters(([list(range(args.T))] + [[args.T-1]]) * num_plots_in_this_figure,
                                       # [v1_ang, v1_ang_pred, v2_ang, v2_ang_pred, v3_ang, v3_ang_pred, v4_ang, v4_ang_pred],
                                       [v_ang[bus_idx // 2] if bus_idx % 2 == 0 else v_ang_pred[bus_idx // 2] for bus_idx in range(plot_id_low,plot_id_high)],
                                       legends_lst[plot_id_low:plot_id_high],
                                       "t (minutes)", voltage_angle_units,
                                       "Test Example {} - {}".format(global_sample_id, angleword),
                                       predict_image_dir,
                                       "Predict_{:03d}_{}{}V.png".format(global_sample_id,extra_string_for_filename, angleword),
                                       extras_dict)
        if tb > 0 and (((tb+1) * testBatchSize) % 200 == 0 or tb==nTestBatches-1):
            ld('Processed {}/{} test examples'.format((tb+1) * testBatchSize, nTest))

    bar.finish()

    if not args.no_prediction_plots:
        ld('Plotted {} prediction examples. All stored to {}'.format(nTest,predict_image_dir))

    # Plot chronological test plot
    extras_dict = {}
    extras_dict["linewidth_list"] = [0.5 * ((i+1) % 2) + 1 for i in range(pop.num_target_measurements)]#[0.5*((i+1) % 2) + 1 for i in range(number_of_nodes_to_estimate * 2)]  # [".", "+"] * (number_of_nodes_to_estimate)
    extras_dict["legend_location"] = 6
    extras_dict['font_size'] = 14
    extras_dict['marker_list'] = ["."] * pop.num_target_measurements
    chronological_plot_mag_list, chronological_plot_ang_list, chronological_legend_list = [], [], []
    for node_id in range(pop.num_nodes):
        chronological_legend_list += [r"$V_{" + str(1 + node_id) + r"}$", r"$\hat V_{" + str(1 + node_id) + r"}$"]
        chronological_plot_mag_list += [chronological_denormalized_ground_truths[:, node_id * 2],
                                         chronological_denormalized_predictions[:,node_id*2]]
        chronological_plot_ang_list += [chronological_denormalized_ground_truths[:, node_id * 2 + 1],
                                         chronological_denormalized_predictions[:, node_id * 2 + 1]]
    # Break the figures into multiple ones
    if not args.no_prediction_plots:
        max_buses_per_figure = 6
        for plot_set in range(int(ceil(pop.num_target_buses / max_buses_per_figure))):
            bus_id_low = plot_set * max_buses_per_figure
            bus_id_high = min(pop.num_target_buses, (plot_set + 1) * max_buses_per_figure)
            plot_id_low = 2 * bus_id_low
            plot_id_high = 2 * bus_id_high
            num_plots_in_this_figure = plot_id_high - plot_id_low
            extra_string_for_filename = "" if bus_id_low == 0 and bus_id_high == pop.num_target_buses else "buses_{}-{}_".format(bus_id_low + 1, bus_id_high)

            for plot_list, unit_word, component_word in zip([chronological_plot_mag_list[plot_id_low:plot_id_high], chronological_plot_ang_list[plot_id_low:plot_id_high]], ["Volts", voltage_angle_units], [magword, angleword]):
                plotManyLines(time_steps_test, plot_list, chronological_legend_list[plot_id_low:plot_id_high], "Time step", unit_word,
                              r"{} (T={}, $\lambda={}$)".format(component_word, args.T, args.equations_regularizer), predict_image_dir,
                              "Chronological_{}{}_T{}.png".format(extra_string_for_filename, component_word, args.T),
                              extras_dict)

    ld("Test (average over all nodes) MSE: {:.3e} MAE: {:.3e}".format(testMSE.mean(),testMAE.mean()))
    for nnode in range(pop.num_target_measurements//pop.num_measurements_per_phasor):
        ld(" - For Re(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, testPerNodeMseVector.mean()[2*nnode], testPerNodeMaeVector.mean()[2*nnode]))
        ld(" - For Im(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, testPerNodeMseVector.mean()[2*nnode+1], testPerNodeMaeVector.mean()[2*nnode+1]))

    ld("Denormalized Test (average over all nodes) MSE: {:.3e} MAE: {:.3e}".format(denormalized_testMSE.mean(),denormalized_testMAE.mean()))
    for nnode in range(pop.num_target_measurements//pop.num_measurements_per_phasor):
        ld(" - For Mag(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, denormalized_testPerNodeMseVector.mean()[2*nnode], denormalized_testPerNodeMaeVector.mean()[2*nnode]))
        ld(" - For Ang(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, denormalized_testPerNodeMseVector.mean()[2*nnode+1], denormalized_testPerNodeMaeVector.mean()[2*nnode+1]))

    # Detailed log prints for MSE and MAE
    MSE_per_example_File.write("Test MSE normalized :\n")
    printPernodeEntry(MSE_per_example_File,"Average",testPerNodeMseVector.mean())
    MSE_per_example_File.write("Test MSE of Re normalized-average-across-buses {:.3e}\n".format(np.mean(testPerNodeMseVector.mean()[0::2])))
    MSE_per_example_File.write("Test MSE of Im normalized-average-across-buses {:.3e}\n".format(np.mean(testPerNodeMseVector.mean()[1::2])))
    MSE_per_example_File.write("Test MSE of All normalized-average-across-buses {:.3e}\n".format(np.mean(testPerNodeMseVector.mean())))
    MSE_per_example_File.write("Test MSE of Magnitude denormalized-average-across-buses {:.3e}\n".format(np.mean(denormalized_testPerNodeMseVector.mean()[0::2])))
    MSE_per_example_File.write("Test MSE of Angle denormalized-average-across-buses {:.3e}\n".format(np.mean(denormalized_testPerNodeMseVector.mean()[1::2])))
    MSE_per_example_File.write("Test MSE of All denormalized-average-across-buses {:.3e}".format(np.mean(denormalized_testPerNodeMseVector.mean())))

    MAE_per_example_File.write("Test MAE normalized :\n")
    printPernodeEntry(MAE_per_example_File,"Average",testPerNodeMaeVector.mean())
    MAE_per_example_File.write("Test MAE of Re normalized-average-across-buses {:.3e}\n".format(np.mean(testPerNodeMaeVector.mean()[0::2])))
    MAE_per_example_File.write("Test MAE of Im normalized-average-across-buses {:.3e}\n".format(np.mean(testPerNodeMaeVector.mean()[1::2])))
    MAE_per_example_File.write("Test MAE of All normalized-average-across-buses {:.3e}\n".format(np.mean(testPerNodeMaeVector.mean())))
    MAE_per_example_File.write("Test MAE of Magnitude denormalized-average-across-buses {:.3e}\n".format(np.mean(denormalized_testPerNodeMaeVector.mean()[0::2])))
    MAE_per_example_File.write("Test MAE of Angle denormalized-average-across-buses {:.3e}\n".format(np.mean(denormalized_testPerNodeMaeVector.mean()[1::2])))
    MAE_per_example_File.write("Test MAE of All denormalized-average-across-buses {:.3e}".format(np.mean(denormalized_testPerNodeMaeVector.mean())))

MSE_per_example_File.close()
MAE_per_example_File.close()

if TensorboardEnabled:
    summary_writer.close()
