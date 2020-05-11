'''
Run multiple trainings & evaluation of the power-flow NNs.
This script runs multiple model trainings to try
different network hyperparameters (aka network depths)
#######################################################
###                                                 ###
###     Be An Accurate Researcher !                 ###
###                                                 ###
###     Before running this script on the results,  ###
###     make sure that you checked *ALL* of the     ###
###     "PHASE I - Configuration" section           ###
###                                                 ###
#######################################################

'''
import pdb
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug
from subprocess import call
import argparse
import os
import itertools
from Utils.logprints import is_log_complete



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running multiple powerflow model trainings and evaluations',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model-name',  choices=['series_voltage', 'dsse'],
                        default='dsse', help='The 1st name in the string')
    parser.add_argument('-model-type', choices=['neuralnet', 'persistent', 'wls'],
                        default='neuralnet',
                        help='Chooses whether the neural network model is trained/evaluated in '
                             'this run, or its non-deep-learning alternatives. '
                             '\"persistent\" completely degenerate the neural network '
                             'model into a model that predicts the voltages by the most recently '
                             'observed voltage. \"wls\" sets the model to be weighted least squares'
                             'estimator of the target values. \'neuralnet\" is the default choice that'
                             'sets the neural network as the model type.')
    parser.add_argument('-dataset-name',
                        choices=['ieee37_smooth_ord_60_downsampling_factor_60','solar_smooth_ord_60_downsampling_factor_60', 'solar_smooth_ord_60', 'solar'],
                        default='ieee37_smooth_ord_60_downsampling_factor_60', help='The 2nd string in the model name.')
    parser.add_argument('-seeds', type=str,
                        default=str(list(range(1,9))).strip(']').strip('[').replace(' ', ''),
                        help='Comma separated list of integer random seeds to perform the training with.')
    parser.add_argument('-epochs', type=int,
                        default=300,
                        help='Number of training epochs applied for each model.')
    parser.add_argument('-lambdas', type=str,
                        default="0.0,0.5,1.0,2.0",
                        help='Comma separated list of coefficients applied to the Power-Flow equation loss component.')
    parser.add_argument('-Ts', type=str,
                        default="5,50",
                        help='Comma separated list of numbers of time steps in the inputs sliding window.')
    parser.add_argument('-Nss', type=str,
                        default="36,35,34,32,28,24,18,12,6,0",
                        help='Comma separated list of numbers of observable power phasors at the last time step.')
    parser.add_argument('-Nvs', type=str,
                        default="0",
                        help='Comma separated list of numbers of observable voltage phasors at the last time step.')
    parser.add_argument('-batch-sizes', type=str,
                        default="50",
                        help='Comma separated list of batch sizes.')
    parser.add_argument('-gpuid', type=int,
                        default=-1,
                        help='The id of the GPU to host the training. The default value of -1 simply '
                             'means that the gpuid will be derived from the seed (by MODing it by 4).')
    parser.add_argument('-logdir', type=str,
                        default="Logs",
                        help='Path to a directory where the log files of the runs should be created.')
    parser.add_argument('--reverse-bus-hiding-order', action="store_true",
                        help='If used, then the Ns or Nv buses will be taken with respect to a reversed'
                             'order of buses')
    parser.add_argument('--slim-auxilliary-network', action="store_true",
                        help='If used, then a slimmer sub-model (Ns-->N/6-->N/6) in charge of '
                             't-1 input (partially observable) is used instead of a Ns-->Ns-->N/6.')
    parser.add_argument('--run-only-if-log-incomplete', action="store_true",
                        help='Use this option if you want to check whether the complete log file already exists in the '
                             'Logs directory. This check will be performed for each run independently and the run will '
                             'be initiated only if there is an incomplete log file."')


    args = parser.parse_args()

    # Some hardcoded values:
    num_gpus = 4
    callee = "python {}.py".format(args.model_name)

    seed_list   = [int(s) for s in args.seeds.replace(' ','').split(",")]
    bs_list     = [int(s) for s in args.batch_sizes.replace(' ', '').split(",")] if args.model_type == 'neuralnet' else [50]
    lambda_list  = [float(s) for s in args.lambdas.replace(' ', '').split(",")] if args.model_type == 'neuralnet' else [0]
    Ts_list = [int(s) for s in args.Ts.replace(' ', '').split(",")]
    Nss_list = [int(s) for s in args.Nss.replace(' ', '').split(",")]
    Nvs_list = [int(s) for s in args.Nvs.replace(' ', '').split(",")]
    num_runs = len(seed_list) * len(lambda_list) * len(bs_list) * len(Ts_list) * len(Nss_list) * len(Nvs_list)
    ld("Performing {} runs...".format(num_runs))

    # Grade-Loop of system calls:
    run_number = 1
    for seed, bs, lambda_, T, Ns, Nv in list(itertools.product(seed_list, bs_list, lambda_list, Ts_list, Nss_list, Nvs_list)):

        ld("Began Run number {}/{}".format(run_number, num_runs))

        model_name_with_params = "{}_{}_{}_seed:{}_bs:{}_lambda:{}_T:{}_Ns:{}_Nv:{}".format(args.model_name, args.dataset_name, args.model_type, seed, bs, lambda_, T, Ns, Nv)
        model_type_str = "-model-type {}".format(args.model_type)
        model_name_str = "-model-name {}".format(model_name_with_params)
        dataset_name_str = "-dataset-name {}".format(args.dataset_name)
        seed_str = "-seed {}".format(seed)
        bs_str = "-batch-size {}".format(bs)
        lambda_str = "-equations-regularizer {}".format(lambda_)
        T_str = "-T {}".format(T)
        Ns_str = "-Ns {}".format(Ns)
        Nv_str = "-Nv {}".format(Nv)
        if args.model_type == 'neuralnet':
            gpuid_str = "-gpuid {}".format(args.gpuid if args.gpuid != -1 else seed % num_gpus)
        else:
            gpuid_str = "-gpuid -1"

        extra_flags_str = "--no-prediction-plots -data-transform standardize --wls-with-power " \
                          "-n-epochs {} --wls-weights-discriminate-hidden " \
                          "-logdir {}".format(args.epochs, args.logdir)
        extra_flags_str = extra_flags_str if not args.reverse_bus_hiding_order else extra_flags_str+" --reverse-bus-hiding-order"
        extra_flags_str = extra_flags_str if not args.slim_auxilliary_network else extra_flags_str+" --slim-auxilliary-network"

        # If a complete filename was found, skip this run.
        log_filename = model_name_with_params + "_test_MSE_per_example.txt"
        if args.run_only_if_log_incomplete and is_log_complete(args.logdir, log_filename):
            ld("Skipping run due to a complete log-file: {}".format(log_filename))
            run_number += 1
            continue
        # Else, run the training
        else:
            caller_list = [callee, model_type_str, model_name_str, dataset_name_str, seed_str, bs_str, lambda_str, T_str, Ns_str, Nv_str, gpuid_str, extra_flags_str]
            caller_string = " ".join(caller_list)
            ld(caller_string)
            os.system(caller_string)

        ld("Finished Run number {}/{}".format(run_number, num_runs))
        run_number += 1
