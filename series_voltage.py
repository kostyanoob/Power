'''

'''
import logging
import tensorflow as tf
import numpy as np
import progressbar
import argparse
import random
import pdb
import sys
import os

from Utils.nn import *
from Utils.plot import plotTwoLines, plotListOfScatters, plotManyLines
from Utils.logprints import *
from Utils.data_transformation import *
from Utils.data_loader import load_dataset
from Utils.complex_numbers import realImagVectorsToMagAngVectors
from Utils.loss import *

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from math import ceil


##################################################################


logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug


parser = argparse.ArgumentParser(description='Powerflow solution estimation from partial observability. '
                                             'This script trains and tests a deep neural network (DNN)  '
                                             'which receives time series of PMU (Phasor Measurment Unit) '
                                             'measurements such as complex power and complex voltage at '
                                             'the buses (nodes) of a distribution power grid. The DNN '
                                             'receives T-1 time steps of full observability followed by and '
                                             'additional time step with partial observability of the powers and voltages. '
                                             'The aim is to complete the voltage estimation at the Ts timestamp.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-model-name', type=str,
                    default='series_voltage',
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
                    default='series_voltage',
                    help='Name of model to be loaded to begin the '
                         'training with (aka to be finetuned)')
parser.add_argument('-dataset-name', type=str,
                    default='solar_smooth_ord_60_downsampling_factor_60',
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
                    default=100,
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
                    default=100,
                    help='Number of PMU measurment time-steps in a single input sequence.')
parser.add_argument('-n-layers-lstm', type=int,
                    default=2,
                    help='Number of LSTM layers stacked')
parser.add_argument('-n-layers-fc', type=int,
                    default=5,
                    help='number of fully connected layers, which follow the LSTM layers.')
parser.add_argument('--polar-complex', action="store_true",
                    help='If specified, this option forces the dataset to be transformed'
                         'from cartesian to polar representation of the complex numbers.')
parser.add_argument('--use-complex-tf-ops', action="store_true",
                    help='If used, then complex data types will be employed inthe computation. '
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
parser.add_argument('--no-trained-scaling', action="store_true",
                    help='If used, then scaling (multiplicative and additive) of the logits will be '
                         'disabled (At the end of the model).')
parser.add_argument('--no-prediction-plots', action="store_true",
                    help='If used, then no per-test-example plots will be generated. Only the '
                         'chronological plot wil be generated in the predictions directory')
parser.add_argument('-seed', type=int, default=42,
                    help='This number sets the initial pseudo-random generators of the numpy '
                         'and of the tensorflow (default:42)')

args = parser.parse_args()

# Main toggles
model_name                  = args.model_name
dataset_dir                 = "Dataset"
dataset_name                = args.dataset_name
training_enabled            = not args.no_training       # if True: then train+saving best model+test occurs. if False: the model is loaded and tested
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
lstm_output_dim             = 12

# Training hyperparameters
nEpochs                     = args.n_epochs
batch_size                  = args.batch_size
dropout_keep_prob           = 1.0
batch_norm_MA_decay         = 0.95      # a fraction for decaying the moving average. Reasonable values for decay are close to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc
momentum                    = 0.9       # a constant momentum for SGD
use_nesterov                = True      # If MomentumOptimizer is used, then it will be a nesterov momentum (
lr_init                     = 0.1       # initial learning rate for SGD
lr_div_epoch_dict           = {40:10, 60:10, 120:10} # epoch:reduction_factor

# Dataset
num_examples_to_generate    = args.num_examples_to_generate # how many examples will be generated from the existing CSV files (generation is carried out via random T-long time series).
test_fraction               = 0.1 # fraction of the generated examples that will become a test set. The splitting between the training and test time series is leakage-safe. Namely, no training time series overlaps with test time series.
validation_fraction         = 0.0 # fraction of the train examples that will become a validation set. Warning: th ecurrent way of splitting the training to validation creates data leakage between the trainin  and validation since the time series overlap!
num_all_measurements        = 16
num_remaining_measurements  = 6
num_hidden_measurements     = 2
num_target_measurements     = 8
#assert(num_all_measurements == num_remaining_measurments + num_missing_measurements)
nValid                      = int(num_examples_to_generate * (1-test_fraction) * validation_fraction) # number of validation examples.
nTest                       = int(num_examples_to_generate * test_fraction) # number of test examples
nTrain                      = num_examples_to_generate - nValid - nTest
testBatchSize               = batch_size #currently degenerated, but could also be different than the training-batch_size...

# Set seeds (except for the tf seed, which will be asserted in the session)
np.random.seed(random_seed)
random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)

if args.polar_complex:
    ld('Error - cannot run with polar complex numbers. This would harm the correctness of the power flow equation solution.')
    sys.exit(-1)

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

dataset_path = os.path.join(dataset_dir,dataset_name)

#####################################################
########## Dataset Preparation ######################
#####################################################
hidden_voltage_bus_id_list = [0, 1, 2, 3]  # hidden from input in T-1 (last) time-step
hidden_power_bus_id_list = [0]  # hidden from input in T-1 (last) time-step
target_voltage_bus_id_list = [0, 1, 2, 3]
target_power_bus_id_list = []
X1_train, X2_train, X2hidden_train, y_train, _,X1_test, X2_test, X2hidden_test, y_test, Y_real_np, Y_imag_np, time_steps_test = load_dataset(dataset_path,
                                                                     test_fraction,
                                                                     num_examples_to_generate,
                                                                     args.T,
                                                                     hidden_voltage_bus_id_list,
                                                                     hidden_power_bus_id_list,
                                                                     target_voltage_bus_id_list,
                                                                     target_power_bus_id_list,
                                                                     num_hidden_measurements_to_keep = num_hidden_measurements,
                                                                     use_polar=args.polar_complex,
                                                                     null_dataset = args.null_dataset,
                                                                     dtype = np_real_dtype)

# # Split the X1,X2,y using the following command.
# X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y,
#                                                                      test_size=test_fraction,
#                                                                      shuffle=True,
#                                                                      random_state=random_seed)
X1_train, X1_validation, X2_train, X2_validation, X2hidden_train, X2hidden_validation, y_train, y_validation = train_test_split(X1_train, X2_train, X2hidden_train, y_train,
                                                                     test_size=validation_fraction,
                                                                     shuffle=True,
                                                                     random_state=random_seed)

# Normalize / standardize
if args.data_transform == 'normalize':
    X1_transformer, X2_transformer, X2hidden_transformer, y_transformer = [NDMinMaxScaler(feature_range=(-1, 1), copy=True), MinMaxScaler((-1, 1)), MinMaxScaler((-1, 1)), MinMaxScaler((-1, 1))]
elif args.data_transform == 'standardize':
    X1_transformer, X2_transformer, X2hidden_transformer, y_transformer = [NDStandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()]
elif args.data_transform == 'none':
    X1_transformer, X2_transformer, X2hidden_transformer, y_transformer = [IdentityScaler(), IdentityScaler(), IdentityScaler(), IdentityScaler()]
#pdb.set_trace()
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

number_of_nodes_to_estimate = y_train.shape[1] //2
assert( num_target_measurements == y_train.shape[1] )
TensorboardEnabled = TensorboardEnabled_valid or TensorboardEnabled_training
assert(not (TensorboardEnabled_valid and TensorboardEnabled_training))

#####################################################
########## Neural Net Construction ##################
#####################################################
graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(random_seed)
    deviceStr = '/cpu:0' if gpuid == -1 else '/gpu:{gpuid}'.format(gpuid=gpuid)
    with tf.device(deviceStr):

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        input_data_X1 = tf.placeholder(tf_real_dtype, shape=(None, args.T-1, num_all_measurements))
        input_data_X2 = tf.placeholder(tf_real_dtype, shape=(None, num_remaining_measurements))
        input_data_X2hidden = tf.placeholder(tf_real_dtype, shape=(None, num_hidden_measurements))
        input_labels  = tf.placeholder(tf_real_dtype, shape=(None, num_target_measurements))
        dropout_prob = tf.placeholder(tf_real_dtype) # keep probability. a real value from an interval [0.0,1.0]
        lr = tf.placeholder(tf_real_dtype)

        # Create the model
        lstm_outputs, final_state = lstm_model(input_data_X1, num_all_measurements, lstm_output_dim, n_layers_lstm, dtype=tf_real_dtype, dropout_keep_prob=dropout_prob)
        auxiliary_inputs = fc_model(input_data_X2, num_remaining_measurements, num_remaining_measurements, 2, dtype=tf_real_dtype) #Non linear transformation 6-->6
        intermediate_features = tf.concat([lstm_outputs[:, -1], auxiliary_inputs], axis=1) # obtain a 524 long vector by concatenating the lstm outputs to the X2 input along dimenstion=1 (dim=0 is the example index in the batch)
        unscaled_logits = fc_model(intermediate_features, lstm_output_dim + num_remaining_measurements, num_target_measurements, n_layers_fc, dtype=tf_real_dtype)
        # some scaler variables (to scale the final normalized logits.
        multiplicative_scaler_vector = fc_model(intermediate_features, lstm_output_dim + num_remaining_measurements, num_target_measurements, 1, activation=tf.nn.leaky_relu, dtype=tf_real_dtype)
        additive_scaler_vector = fc_model(intermediate_features, lstm_output_dim + num_remaining_measurements, num_target_measurements, 1,  activation=tf.nn.leaky_relu, dtype=tf_real_dtype)
        scaled_logits = tf.math.add(tf.math.multiply(unscaled_logits, multiplicative_scaler_vector), additive_scaler_vector) if not args.no_trained_scaling else unscaled_logits

        # Model setting according to command line choice
        if args.model_type == 'neuralnet':
            logits = scaled_logits
        elif args.model_type == 'persistent':
            logits = input_data_X1[:,-1,number_of_nodes_to_estimate*2:]
        elif args.model_type == 'wls':
            # TODO implement WLS here
            pass
        else:
            ld("Error, unknown model type \"{}\"".format(args.model_type))


        # Define regularization on all the weights, but not the biases
        L2_norms_of_all_weights_list = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]

        # Power flow equation as a regularizer:
        #  s: complex power vectors (batch_size x n_nodes x 1)
        #  v: complex voltage vectors of the ground truth voltages (batch_size x n_nodes x 1)
        #  v_est: complex voltage vectors of the estimated voltages (batch_size x n_nodes x 1)
        #  Y: complex admittance matrix (n_nodes x n_nodes)
        real_indices = np.array(range(0, num_target_measurements - 1,2))
        imag_indices = np.array(range(1, num_target_measurements, 2))
        # Inverse transform the voltages and the powers.
        denormalized_input_data_X2hidden = tf_inverse_transform(X2hidden_transformer, input_data_X2hidden, tf_real_dtype)
        denormalized_input_data_X2 = tf_inverse_transform(X2_transformer, input_data_X2, tf_real_dtype)
        s_real_imag_interleaved = tf.concat([denormalized_input_data_X2hidden, denormalized_input_data_X2], axis=1) # hidden (S1 bus) comes before the rest X2
        s_real = tf.gather(s_real_imag_interleaved, real_indices, axis=1)
        s_imag = tf.gather(s_real_imag_interleaved, imag_indices, axis=1)
        denormalized_input_labels = tf_inverse_transform(y_transformer, input_labels, tf_real_dtype)
        v_real = tf.gather(denormalized_input_labels, real_indices, axis=1)
        v_imag = tf.gather(denormalized_input_labels, imag_indices, axis=1)
        denormalized_logits = tf_inverse_transform(y_transformer, logits, tf_real_dtype)
        v_est_real = tf.gather(denormalized_logits, real_indices, axis=1)
        v_est_imag = tf.gather(denormalized_logits, imag_indices, axis=1)
        Y_real = tf.constant(Y_real_np, dtype=tf_real_dtype)
        Y_imag = tf.constant(Y_imag_np, dtype=tf_real_dtype)
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


        regularizing_term = args.weight_decay * tf.add_n(L2_norms_of_all_weights_list) + \
                            args.equations_regularizer * pfe_loss
        # print_op = tf.print([s_real[0],s_imag[0], v_real[0], v_imag[0], Y_real, Y_imag, pfe_numeric_error])
        # Loss function and output of the network
        #logits = tf.Print(logits, [tf.shape(input_labels)], "\nInput labels shape:")
        #logits = tf.Print(logits, [tf.shape(logits)], "\nLogits shape:")
        per_node_mse = tf.reduce_mean(tf.square(tf.subtract(input_labels, logits)),axis=0)
        per_node_mae = tf.reduce_mean(tf.math.abs(tf.subtract(input_labels, logits)),axis=0)

        mse = tf.reduce_mean(per_node_mse)
        mae = tf.reduce_mean(per_node_mae)
        loss = mse + regularizing_term
        predictions = logits

        # Optimizer creation
        optimizer = tf.train.AdamOptimizer()

        # Gradients: computation, clipping (to avoid exploding gradients) and application thereof (to update the weights)
        grads_unclipped = optimizer.compute_gradients(loss, tf.trainable_variables())
        grads = [(tf.clip_by_value(gv[0], -1, 1), gv[1]) for gv in grads_unclipped]
        optimizer = optimizer.apply_gradients(grads)

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
            for nnode in range(number_of_nodes_to_estimate):
                tf.summary.scalar('ReV{}'.format(nnode+1), per_node_mse[2*nnode])
                tf.summary.scalar('ImV{}'.format(nnode + 1), per_node_mse[2 * nnode + 1])
        with tf.name_scope('Nodewise_MAE'):
            for nnode in range(number_of_nodes_to_estimate):
                tf.summary.scalar('ReV{}'.format(nnode+1), per_node_mae[2*nnode])
                tf.summary.scalar('ImV{}'.format(nnode + 1), per_node_mae[2 * nnode + 1])
        with tf.name_scope('Nodewise_Denormalized_MSE'):
            for nnode in range(number_of_nodes_to_estimate):
                tf.summary.scalar('ReV{}'.format(nnode+1), per_node_denormalized_mse[2*nnode])
                tf.summary.scalar('ImV{}'.format(nnode + 1), per_node_denormalized_mse[2 * nnode + 1])
        with tf.name_scope('Nodewise_Denormalized_MAE'):
            for nnode in range(number_of_nodes_to_estimate):
                tf.summary.scalar('ReV{}'.format(nnode+1), per_node_denormalized_mae[2*nnode])
                tf.summary.scalar('ImV{}'.format(nnode + 1), per_node_denormalized_mae[2 * nnode + 1])

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

lr_current = lr_init

if training_enabled:
    with tf.Session(graph=graph, config=tfcfg) as session:

        if args.revert_to_single_precision:
            ld("Using SINGLE precision.")
        else:
            ld("Using DOUBLE precision.")

        saver = tf.train.Saver()
        if restore_session:
            actionTaken = "Loaded existing "
            saver.restore(session, "Models/" + model_name_for_loading + ".ckpt")
        else:  
            actionTaken = "Initialized new "
            session.run(init_op)
        nParams = countParams()

        ld(actionTaken + "model, consisting of " + str(nParams) + " parameters. Beginning training...")
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
        trainPerNodeMses = np.zeros((nEpochs,2*number_of_nodes_to_estimate))
        validPerNodeMses = np.zeros((nEpochs,2*number_of_nodes_to_estimate))

        # Open Log File
        logFile = open(os.path.join(log_outputdir, model_name+".txt"), "w")
        MSE_File = open(os.path.join(log_outputdir, model_name + "_test_MSE.txt"), "w")
        printLogHeader(logFile, nValid > 0)
        printPernodeMSEHeader(MSE_File, ["ReV1","ImV1","ReV2","ImV2","ReV3","ImV3","ReV4","ImV4"])
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

            # Maintain learning rate decay mechanism
            if epoch in lr_div_epoch_dict:
                lr_current /= lr_div_epoch_dict[epoch]

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
                             dropout_prob: dropout_keep_prob,
                             lr: lr_current}

                if TensorboardEnabled_training:
                    summary_str, _, batch_loss, batch_mse, batch_pernode_loss = session.run([summary, optimizer, loss, mse, per_node_mse],
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
            if minLoss < validLosses[epoch]:
                saver.save(session, "Models/" + model_name + "_best.ckpt")
                minLoss = validLosses[epoch]
            val_test_str = " validation " if nValid > 0 else " test"
            ld("Epoch " + str(epoch) + val_test_str +
               " loss {:3.6f} mse {:.3e} pfee {:.3e} pfenmse {:.3e} pfenmae {:.3e}"
               .format(validLosses[epoch], validMSE[epoch], validPFEE[epoch], validPFENMSE[epoch], validPFENMAE[epoch]))
            for nnode in range(number_of_nodes_to_estimate):
                ld(" - Test MSE of Re(V{}): {:.8f}".format(nnode + 1, validPerNodeMses[epoch, 2 * nnode]))
                ld(" - Test MSE of Im(V{}): {:.8f}".format(nnode + 1, validPerNodeMses[epoch, 2 * nnode + 1]))

            # Maintain log file after each epoch
            # Must normalize the training measurements, since they were summed over all the batches
            trainLosses[epoch]   /= nBatchesPerEpoch
            printLogEntry(logFile, logLine, validLosses[epoch], trainLosses[epoch], validMSE[epoch], trainMSE[epoch])
            printPernodeEntry(MSE_File, logLine, validPerNodeMses[epoch])
            logLine += 1

            # In case when it's a full-data (no validation) run - save the model every 10 epochs
            if epoch % 10 == 9 or epoch == (nEpochs - 1):
                saver.save(session, "Models/"+model_name+".ckpt")

        # Training complete
        bar.finish()


        # Calculate Test Accuracy for the BEST MODEL (requires restoration of the best saved session)
        if nValid > 0:
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

        for tb in range(nTestBatches):
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

            # Denormalize
            # X1_test[sample_low:sample_low+predictions_t.shape[0]] = X1_transformer.inverse_transform(X1_test[sample_low:sample_low+predictions_t.shape[0]])
            # predictions_t = y_transformer.inverse_transform(predictions_t)
            # y_test[sample_low:sample_low+predictions_t.shape[0]] = y_transformer.inverse_transform(y_test[sample_low:sample_high])
            denormalized_labels = y_transformer.inverse_transform(y_test[sample_low:sample_high])
            denormalized_predictions = y_transformer.inverse_transform(predictions_t)
            real_indices = np.array(range(0, num_target_measurements - 1, 2))
            imag_indices = np.array(range(1, num_target_measurements, 2))
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

        # Create plots of (train, validation) if validation exists, else, (train, test)
        validStr = 'Validation' if nValid > 0 else 'Test'
        plotTwoLines(trainLosses, validLosses, range(1,nEpochs+1),
                    "Epoch", "Loss", validStr + " Loss", figures_outputdir, model_name+"_Loss.jpg",
                    isAnnotatedMin=False, validStr=validStr)
        plotTwoLines(trainMSE, validMSE, range(1,nEpochs+1),
                    "Epoch", "MSE(%)", "Mean Squared Error (Test=%.8f" % testMSE.mean() + ")", figures_outputdir, model_name + "_MSE.jpg",
                    isAnnotatedMin=True, validStr=validStr)

        ld("Test (average over all nodes) MSE: {:.3e} MAE: {:.3e}".format(testMSE.mean(),testMAE.mean()))
        for nnode in range(number_of_nodes_to_estimate):
            ld(" - For Re(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, testPerNodeMseVector.mean()[2*nnode], testPerNodeMaeVector.mean()[2*nnode]))
            ld(" - For Im(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, testPerNodeMseVector.mean()[2*nnode+1], testPerNodeMaeVector.mean()[2*nnode+1]))
        ld("Denormalized Test (average over all nodes) MSE: {:.3e} MAE: {:.3e}".format(denormalized_testMSE.mean(),denormalized_testMAE.mean()))
        for nnode in range(number_of_nodes_to_estimate):
            ld(" - For Mag(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, denormalized_testPerNodeMseVector.mean()[2*nnode], denormalized_testPerNodeMaeVector.mean()[2*nnode]))
            ld(" - For Ang(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, denormalized_testPerNodeMseVector.mean()[2*nnode+1], denormalized_testPerNodeMaeVector.mean()[2*nnode+1]))


        testLossStr = "Test MSE: normalized: {:.20f} denormalized {:.20f}".format(testMSE.mean(),denormalized_testMSE.mean())
        ld(testLossStr)
        logFile.write(testLossStr)
        logFile.close()
        MSE_File.close()
        ld("************************************************************")

        # Final model saving (after all the trainings)
        saver.save(session, "Models/"+model_name+".ckpt")



ld("Model Testing")
with tf.Session(graph=graph, config=tfcfg) as session:
    # Restore variables from disk.

    saver = tf.train.Saver()
    saver.restore(session, "Models/" + model_name + ".ckpt")
    nParams = countParams()
    ld("Model restored from Models/" + model_name + ".ckpt, having " + str(nParams) + " parameters")

    nTestBatches = int(ceil(nTest / float(testBatchSize)))
    testLoss, testMSE, testMAE = AverageTracker(0.0), AverageTracker(0.0), AverageTracker(0.0)
    testPerNodeMseVector, testPerNodeMaeVector = AverageTracker(np.zeros_like(y_test[0])), AverageTracker(np.zeros_like(y_test[0]))
    denormalized_testMSE, denormalized_testMAE = AverageTracker(0.0), AverageTracker(0.0)
    denormalized_testPerNodeMseVector, denormalized_testPerNodeMaeVector = AverageTracker(np.zeros_like(y_test[0])), AverageTracker(np.zeros_like(y_test[0]))
    chronological_denormalized_ground_truths = np.zeros((time_steps_test.shape[0], num_target_measurements), dtype=np_real_dtype)
    chronological_denormalized_predictions = np.zeros((time_steps_test.shape[0], num_target_measurements), dtype=np_real_dtype)

    predict_image_dir = os.path.join("Figures", "Predictions_" + model_name)
    if not os.path.exists(predict_image_dir):
        os.makedirs(predict_image_dir)

    for tb in range(nTestBatches):
        if tb > 0 and (tb*testBatchSize) % 200 == 0:
            ld('Processed {}/{} test examples'.format(tb*testBatchSize, nTest))
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
            # import pdb
            # pdb.set_trace()

            v1_mag = X1_test[global_sample_id, :, 8].tolist() + [y_test[global_sample_id, 0]]
            v2_mag = X1_test[global_sample_id, :, 10].tolist() + [y_test[global_sample_id, 2]]
            v3_mag = X1_test[global_sample_id, :, 12].tolist() + [y_test[global_sample_id, 4]]
            v4_mag = X1_test[global_sample_id, :, 14].tolist() + [y_test[global_sample_id, 6]]
            v1_mag_pred = [predictions_t[i, 0]]
            v2_mag_pred = [predictions_t[i, 2]]
            v3_mag_pred = [predictions_t[i, 4]]
            v4_mag_pred = [predictions_t[i, 6]]
            v1_phase = X1_test[global_sample_id, :, 9].tolist() + [y_test[global_sample_id, 1]]
            v2_phase = X1_test[global_sample_id, :, 11].tolist() + [y_test[global_sample_id, 3]]
            v3_phase = X1_test[global_sample_id, :, 13].tolist() + [y_test[global_sample_id, 5]]
            v4_phase = X1_test[global_sample_id, :, 15].tolist() + [y_test[global_sample_id, 7]]
            v1_phase_pred = [predictions_t[i, 1]]
            v2_phase_pred = [predictions_t[i, 3]]
            v3_phase_pred = [predictions_t[i, 5]]
            v4_phase_pred = [predictions_t[i, 7]]

            # Force magnitude-angle representaton for the voltages
            magword = "Magnitude"
            phaseword = "Phase"
            voltage_phase_units = "Degrees"
            if not args.polar_complex:
                v1_mag, v1_phase = realImagVectorsToMagAngVectors(v1_mag, v1_phase)
                v2_mag, v2_phase = realImagVectorsToMagAngVectors(v2_mag, v2_phase)
                v3_mag, v3_phase = realImagVectorsToMagAngVectors(v3_mag, v3_phase)
                v4_mag, v4_phase = realImagVectorsToMagAngVectors(v4_mag, v4_phase)
                v1_mag_pred, v1_phase_pred = realImagVectorsToMagAngVectors(v1_mag_pred, v1_phase_pred)
                v2_mag_pred, v2_phase_pred = realImagVectorsToMagAngVectors(v2_mag_pred, v2_phase_pred)
                v3_mag_pred, v3_phase_pred = realImagVectorsToMagAngVectors(v3_mag_pred, v3_phase_pred)
                v4_mag_pred, v4_phase_pred = realImagVectorsToMagAngVectors(v4_mag_pred, v4_phase_pred)

            ground_truths = np.array([v1_mag[args.T-1], v1_phase[args.T-1], v2_mag[args.T-1], v2_phase[args.T-1],
                                      v3_mag[args.T-1], v3_phase[args.T-1], v4_mag[args.T-1], v4_phase[args.T-1]],
                                    dtype=np_real_dtype)
            predicteds = np.array(v1_mag_pred + v1_phase_pred + v2_mag_pred + v2_phase_pred +
                                  v3_mag_pred + v3_phase_pred + v4_mag_pred + v4_phase_pred,
                                    dtype=np_real_dtype)

            chronological_denormalized_ground_truths[global_sample_id, :] = ground_truths
            chronological_denormalized_predictions[global_sample_id, :] = predicteds

            denormalized_NumpyPerNodeMseVector = se_func(ground_truths, predicteds)
            denormalized_NumpyPerNodeMaeVector = ae_func(ground_truths, predicteds)
            denormalized_testMSE.update(np.mean(denormalized_NumpyPerNodeMseVector))
            denormalized_testPerNodeMseVector.update(denormalized_NumpyPerNodeMseVector)
            denormalized_testMAE.update(np.mean(denormalized_NumpyPerNodeMaeVector))
            denormalized_testPerNodeMaeVector.update(denormalized_NumpyPerNodeMaeVector)

            # Set up neat appearance for the plot.
            extras_dict = {}
            # magword = "Magnitude" if args.polar_complex else "Real"
            # phaseword = "Phase" if args.polar_complex else "Imaginary"
            # voltage_phase_units = "Degrees" if args.polar_complex else "Volts"
            #extras_dict["marker_list"] = [".","+"]*(number_of_nodes_to_estimate)
            from matplotlib.text import TextPath
            # from matplotlib.font_manager import FontProperties
            # font_entry = FontProperties(family='sans-serif',size='xx-large')
            # TextPath((0, 0), r"$.\hat V_{}$".format(i // 2 + 1), prop=font_entry, size=10000, usetex=True)
            if not args.no_prediction_plots:
                from matplotlib.markers import MarkerStyle
                extras_dict["marker_list"] = [MarkerStyle(marker=".", fillstyle='none') if i%2==0 else MarkerStyle(marker="+", fillstyle='full') for i in range(number_of_nodes_to_estimate*2)]#[".", "+"] * (number_of_nodes_to_estimate)
                extras_dict["marker_scaler_list"] = [40, 20] * (number_of_nodes_to_estimate)
                extras_dict["legend_location"] = 6
                extras_dict['font_size'] = 14
                legends_lst = []
                for jk in range(number_of_nodes_to_estimate):
                    legends_lst += [r"$V_{}(0:{})$".format(1+jk,args.T - 1), r"$\hat V_{}({})$".format(1+jk,args.T - 1)]

                plotListOfScatters(([list(range(args.T))] + [[args.T-1]])*(number_of_nodes_to_estimate),
                                   [v1_mag, v1_mag_pred, v2_mag, v2_mag_pred,v3_mag, v3_mag_pred, v4_mag, v4_mag_pred],
                                   legends_lst,
                                   "t (minutes)", "Volts",
                                   "Test Example {} - {}".format(global_sample_id, magword),
                                   predict_image_dir,
                                   "Predict_{:03d}_{}_V.png".format(global_sample_id, magword),
                                   extras_dict)
                plotListOfScatters(([list(range(args.T))] + [[args.T-1]])*(number_of_nodes_to_estimate),
                                   [v1_phase, v1_phase_pred, v2_phase, v2_phase_pred, v3_phase, v3_phase_pred, v4_phase, v4_phase_pred],
                                   legends_lst,
                                   "t (minutes)", voltage_phase_units,
                                   "Test Example {} - {}".format(global_sample_id, phaseword),
                                   predict_image_dir,
                                   "Predict_{:03d}_{}_V.png".format(global_sample_id, phaseword),
                                   extras_dict)
    if not args.no_prediction_plots:
        ld('Plotted {} prediction examples. All stored to {}'.format(nTest,predict_image_dir))

    # Plot chronological test plot
    extras_dict = {}
    extras_dict["linewidth_list"] = [0.5*((i+1) % 2) + 1 for i in range(number_of_nodes_to_estimate * 2)]#[0.5*((i+1) % 2) + 1 for i in range(number_of_nodes_to_estimate * 2)]  # [".", "+"] * (number_of_nodes_to_estimate)
    extras_dict["legend_location"] = 6
    extras_dict['font_size'] = 14
    chronological_plot_mag_list, chronological_plot_ang_list, chronological_legend_list = [], [], []
    for node_id in range(number_of_nodes_to_estimate):
        chronological_legend_list += [r"$V_{}$".format(1 + node_id), r"$\hat V_{}$".format(1 + node_id)]
        chronological_plot_mag_list += [chronological_denormalized_ground_truths[:, node_id * 2],
                                         chronological_denormalized_predictions[:,node_id*2]]
        chronological_plot_ang_list += [chronological_denormalized_ground_truths[:, node_id * 2 + 1],
                                         chronological_denormalized_predictions[:, node_id * 2 + 1]]
    for plot_list, unit_word, component_word in zip([chronological_plot_mag_list, chronological_plot_ang_list], ["Volts", voltage_phase_units], [magword, phaseword]):
        plotManyLines(time_steps_test, plot_list, chronological_legend_list, "Time step", unit_word,
                      "{} (T={})".format(component_word, args.T), predict_image_dir,
                      "Chronological_{}_T{}.png".format(component_word, args.T),
                      extras_dict)

    ld("Test (average over all nodes) MSE: {:.3e} MAE: {:.3e}".format(testMSE.mean(),testMAE.mean()))
    for nnode in range(number_of_nodes_to_estimate):
        ld(" - For Re(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, testPerNodeMseVector.mean()[2*nnode], testPerNodeMaeVector.mean()[2*nnode]))
        ld(" - For Im(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, testPerNodeMseVector.mean()[2*nnode+1], testPerNodeMaeVector.mean()[2*nnode+1]))

    ld("Denormalized Test (average over all nodes) MSE: {:.3e} MAE: {:.3e}".format(denormalized_testMSE.mean(),denormalized_testMAE.mean()))
    for nnode in range(number_of_nodes_to_estimate):
        ld(" - For Mag(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, denormalized_testPerNodeMseVector.mean()[2*nnode], denormalized_testPerNodeMaeVector.mean()[2*nnode]))
        ld(" - For Ang(V{}): Test MSE {:.3e}; Test MAE {:.3e}".format(nnode + 1, denormalized_testPerNodeMseVector.mean()[2*nnode+1], denormalized_testPerNodeMaeVector.mean()[2*nnode+1]))


if TensorboardEnabled:
    summary_writer.close()
