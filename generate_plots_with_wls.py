"""
A script that generates multiple plots from log files of
the dsse.py runs (produced by the model_run_script.py)
"""

import numpy as np
import pandas as pd
from Utils.plot import plotManyLines
from Utils.logprints import append_df_to_excel
import itertools
import os
import argparse
import logging
import xlsxwriter
import pdb
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug

from Utils.logprints import is_log_test_MSE_per_example_complete

escape_dict={'\a':r'\a',
           '\b':r'\b',
           '\c':r'\c',
           '\f':r'\f',
           '\n':r'\n',
           '\r':r'\r',
           '\t':r'\t',
           '\v':r'\v',
           '\'':r'\'',
           '\"':r'\"',
           '\0':r'\0',
           '\1':r'\1',
           '\2':r'\2',
           '\3':r'\3',
           '\4':r'\4',
           '\5':r'\5',
           '\6':r'\6',
           '\7':r'\7',
           '\8':r'\8',
           '\9':r'\9'}


def logfile_test_MSE_per_example_MSEs(dir, log_filename):
    """
    Reads a logfile and returns the two MSE values written at the last line thereof.
    :param dir: a directory where all the log-files reside
    :param log_filename: the filename of the log
    :return: a 6-tuple which contains:
    mse_mag, mse_ang, mse, denormalized_mse_mag, denormalized_mse_ang, denormalized_mse

    Note, this function parses the test_MSE per example log file,
    which is normally terminated by the following 6 lines:
    Test MSE of Re normalized-average-across-buses 3.629e-02
    Test MSE of Im normalized-average-across-buses 5.467e-03
    Test MSE of All normalized-average-across-buses 2.088e-02
    Test MSE of Magnitude denormalized-average-across-buses 3.672e-02
    Test MSE of Angle denormalized-average-across-buses 1.702e+01
    Test MSE of All denormalized-average-across-buses 8.528e+00

    """
    if not is_log_test_MSE_per_example_complete(dir, log_filename):
        return None

    with open(os.path.join(dir, log_filename),"r") as txtFile:
        last_6_lines = txtFile.readlines()[-6:]
        mse_list = []
        for log_line in last_6_lines:
            split_line = log_line.split(" ")
            mse_list.append(float(split_line[-1]))
        # mse_mag = float(split_line[-1])
        # mse_ang = float(split_line[-1])
        # mse = float(split_line[-1])
        # denormalized_mse_mag = float(split_line[-1])
        # denormalized_mse_ang = float(split_line[-1])
        # denormalized_mse = float(split_line[-1])

    # The mse_list[0..5] contains:
    # mse_mag, mse_ang, mse, denormalized_mse_mag, denormalized_mse_ang, denormalized_mse
    return mse_list


def raw(text):
    """Returns a raw string representation of text"""
    new_string=''
    for char in text:
        try: new_string+=escape_dict[char]
        except KeyError: new_string+=char
    return new_string

parser = argparse.ArgumentParser(description='Generating loss and accuracy plots of multiple models,'
                                             ' which were ran with multiple seeds (for statistic '
                                             'stability) and produces aggregated plots.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# the filenames are constructed as follows:
# <model_name>.txt
parser.add_argument('-input-data-folder', type=str, metavar='<str>',
                    default="Logs",
                    help='The path to the logs files directory')
parser.add_argument('-output-directory', type=str, metavar='<str>',
                    default="Figures",
                    help='The path where the output images should be written to '),
parser.add_argument('-model-name-list', type=str, metavar='<str>',
                    default='dsse',
                    help='Comma separated names of the model names, that should be aggregated.')
parser.add_argument('-model-type-list', type=str, metavar='<str>',
                    default='neuralnet,wls',
                    help='Comma separated types of the model names, that should be aggregated.')
parser.add_argument('-non-parametric-model-name-list', type=str, metavar='<str>',
                    default='persistent',
                    help='Comma separated names of the model names, which were run without any parameters.')
parser.add_argument('-datasets', type=str, metavar='<str>',
                    default="ieee37_smooth_ord_60_downsampling_factor_60",
                    help='Datsets, comma separated strings')
parser.add_argument('-batch-sizes', type=str, metavar='<str>',
                    default="50",
                    help='batch sizes with which were performed, comma separated integers')
parser.add_argument('-seeds', type=str,
                    default="1,2,3,4,5,6,7,8", metavar='<str>',
                    help='random seeds with which the runs were performed, comma separated integers')
parser.add_argument('-Ts', type=str,
                    default="5,50",
                    help='Comma separated list of numbers of time steps in the inputs sliding window.')
parser.add_argument('-Nss', type=str,
                    default="36,35,34,32,28,24,18,12,6,0",
                    help='Comma separated list of numbers of time steps in the inputs sliding window.')
parser.add_argument('-Nvs', type=str,
                    default="0",
                    help='Comma separated list of numbers of time steps in the inputs sliding window.')
parser.add_argument('-lambdas', type=str,
                    default="0.0,0.5,1.0,2.0",
                    help='Comma separated list of coefficients applied to the Power-Flow equation loss component.')
parser.add_argument('-plot_group_list', type=str, metavar='<str>',
                    default="Power_Observability_Sweep",
                    help='Comma separated names of plots to be generated.')
parser.add_argument('-legend-x', type=float, metavar='<str>',
                    default=-1,
                    help='X-position of the legends (in [0.0,1.0]). '
                         'A default value of -1 sets the legend positioning to automatically "best"')
parser.add_argument('-legend-y', type=float, metavar='<str>',
                    default=-1,
                    help='Y-position of the legends (in [0.0,1.0]). '
                         'A default value of -1 sets the legend positioning to automatically "best"')
parser.add_argument('--keep-std-bars', action="store_true",
                    help='Enable vertical lines indicating the std of ecery data point in the plot')
args = parser.parse_args()

model_name_list = args.model_name_list.split(",")
non_parametric_model_name_list = args.non_parametric_model_name_list.split(",") if args.non_parametric_model_name_list != '{}' else []
plot_group_list = args.plot_group_list.split(",")
datasets = args.datasets.split(",")
model_type_list = args.model_type_list.split(",")
seeds = [int(x) for x in args.seeds.split(",")]
batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
lambdas = [float(x) for x in args.lambdas.split(",")]
Ts = [int(x) for x in args.Ts.split(",")]
Nss = [int(x) for x in args.Nss.split(",")]
Nvs = [int(x) for x in args.Nvs.split(",")]


if not os.path.exists(args.output_directory):
    os.makedirs(args.output_directory)

titleStr = ""#"Sparsity in SimpleNet weights during training"
xtitleStr = "Ns"

###############################################################
################     Plot of lambda_T      ####################
###############################################################
outfile_prefix = "Power_Observability_Sweep"
outfile_excel_path = os.path.join(args.output_directory, outfile_prefix + ".xlsx")

if outfile_prefix in plot_group_list:
    ld("Running {} result aggregation!".format(outfile_prefix))


    for model_name, dataset_name, batch_size, Nv in list(itertools.product(model_name_list, datasets, batch_sizes, Nvs)):

        model_name_dataset = "{}_{}".format(model_name, dataset_name)
        nesterov = "False"
        Result_mag_mean_dict, Result_ang_mean_dict, Result_mean_dict = {}, {}, {}
        Result_mag_min_dict, Result_ang_min_dict, Result_min_dict = {}, {}, {}
        Result_mag_max_dict, Result_ang_max_dict, Result_max_dict = {}, {}, {}
        Result_mag_std_dict, Result_ang_std_dict, Result_std_dict = {}, {}, {}
        Legend_dict = {}
        plot_is_ready_for_creation = True
        for model_type, lambda_, T, Ns in list(itertools.product(model_type_list, lambdas, Ts, Nss)):

            if model_type == "neuralnet":
                concatenated_parameters = "{}_{}_{}_{}".format(model_type, lambda_, T, Ns)
                legend_key = "{}_{}_{}".format(model_type, lambda_, T)
                Legend_dict[legend_key] = r"$DNN(T={},\lambda={})$".format(T,lambda_)
            elif model_type == "wls":
                lambda_ = int(0) # The WLS models do not generally have lambda coefficient.
                concatenated_parameters = "{}_{}_{}_{}".format(model_type, lambda_, T, Ns)
                legend_key = "{}_{}_{}".format(model_type, lambda_, T)
                # Ugly hack:
                if concatenated_parameters in Result_mean_dict:
                    continue

                Legend_dict[legend_key] = r"$WLS(T={})$".format(T)
                    # Accumulate the current configuration with all the differently sed runs
            for seedid, seed in enumerate(seeds):

                log_filename = "{}_{}_seed:{}_bs:{}_lambda:{}_T:{}_Ns:{}_Nv:{}_test_MSE_per_example.txt".format(model_name_dataset,
                                                                                           model_type,
                                                                                           seed, batch_size,
                                                                                           lambda_, T, Ns, Nv)
                mse_and_denormalizedMse = logfile_test_MSE_per_example_MSEs(args.input_data_folder, log_filename)

                if mse_and_denormalizedMse is None:
                    ld('Encountered incomplete or non existing log file {}'.format(log_filename))
                    plot_is_ready_for_creation = False
                    break
                else:
                    _, _, _, denormalized_mse_mag, denormalized_mse_ang, denormalized_mse = mse_and_denormalizedMse

                if not concatenated_parameters in Result_mean_dict:
                    # MAG-MSE
                    Result_mag_std_dict[concatenated_parameters] = np.zeros(len(seeds), dtype=np.float64)
                    Result_mag_std_dict[concatenated_parameters][seedid] = denormalized_mse_mag
                    Result_mag_mean_dict[concatenated_parameters] = denormalized_mse_mag
                    Result_mag_min_dict[concatenated_parameters] = denormalized_mse_mag
                    Result_mag_max_dict[concatenated_parameters] = denormalized_mse_mag
                    # ANG-MSE
                    Result_ang_std_dict[concatenated_parameters] = np.zeros(len(seeds), dtype=np.float64)
                    Result_ang_std_dict[concatenated_parameters][seedid] = denormalized_mse_ang
                    Result_ang_mean_dict[concatenated_parameters] = denormalized_mse_ang
                    Result_ang_min_dict[concatenated_parameters] = denormalized_mse_ang
                    Result_ang_max_dict[concatenated_parameters] = denormalized_mse_ang
                    # MSE
                    Result_std_dict[concatenated_parameters] = np.zeros(len(seeds), dtype=np.float64)
                    Result_std_dict[concatenated_parameters][seedid] = denormalized_mse
                    Result_mean_dict[concatenated_parameters] = denormalized_mse
                    Result_min_dict[concatenated_parameters] = denormalized_mse
                    Result_max_dict[concatenated_parameters] = denormalized_mse
                else:
                    # MAG-MSE
                    Result_mag_std_dict[concatenated_parameters][seedid] = denormalized_mse_mag
                    Result_mag_min_dict[concatenated_parameters] = min(Result_mag_min_dict[concatenated_parameters], denormalized_mse_mag)
                    Result_mag_max_dict[concatenated_parameters] = np.maximum(Result_mag_max_dict[concatenated_parameters], denormalized_mse_mag)
                    Result_mag_mean_dict[concatenated_parameters] += denormalized_mse_mag
                    # ANG-MSE
                    Result_ang_std_dict[concatenated_parameters][seedid] = denormalized_mse_ang
                    Result_ang_min_dict[concatenated_parameters] = min(Result_ang_min_dict[concatenated_parameters], denormalized_mse_ang)
                    Result_ang_max_dict[concatenated_parameters] = np.maximum(Result_ang_max_dict[concatenated_parameters], denormalized_mse_ang)
                    Result_ang_mean_dict[concatenated_parameters] += denormalized_mse_ang
                    # MSE
                    Result_std_dict[concatenated_parameters][seedid] = denormalized_mse
                    Result_min_dict[concatenated_parameters] = min(Result_min_dict[concatenated_parameters], denormalized_mse)
                    Result_max_dict[concatenated_parameters] = np.maximum(Result_max_dict[concatenated_parameters], denormalized_mse)
                    Result_mean_dict[concatenated_parameters] += denormalized_mse

            # Average out the seed runs
            if plot_is_ready_for_creation:
                Result_mag_mean_dict[concatenated_parameters] /= len(seeds)
                Result_mag_std_dict[concatenated_parameters] = np.std(Result_mag_std_dict[concatenated_parameters], axis=0)
                Result_ang_mean_dict[concatenated_parameters] /= len(seeds)
                Result_ang_std_dict[concatenated_parameters] = np.std(Result_ang_std_dict[concatenated_parameters], axis=0)
                Result_mean_dict[concatenated_parameters] /= len(seeds)
                Result_std_dict[concatenated_parameters] = np.std(Result_std_dict[concatenated_parameters], axis=0)
            else:
                break

        # Check out all the non-parametric models
        if plot_is_ready_for_creation:

            Result_mag_nonparametric_mean_dict, Result_ang_nonparametric_mean_dict, Result_nonparametric_mean_dict = {}, {}, {}
            Result_mag_nonparametric_min_dict, Result_ang_nonparametric_min_dict, Result_nonparametric_min_dict = {}, {}, {}
            Result_mag_nonparametric_max_dict, Result_ang_nonparametric_max_dict, Result_nonparametric_max_dict = {}, {}, {}
            Result_mag_nonparametric_std_dict, Result_ang_nonparametric_std_dict, Result_nonparametric_std_dict = {}, {}, {}
            Legend_nonparametric_dict = {}
            for non_parametric_model_name in non_parametric_model_name_list:
                non_parametric_model_name_dataset = "{}_{}".format(non_parametric_model_name, dataset_name)
                log_filename = "{}_test_MSE_per_example.txt".format(non_parametric_model_name_dataset)
                mse_and_denormalizedMse = logfile_test_MSE_per_example_MSEs(args.input_data_folder, log_filename)
                if mse_and_denormalizedMse is None:
                    plot_is_ready_for_creation = False
                    break
                else:
                    _, _, _, denormalized_mse_mag, denormalized_mse_ang, denormalized_mse = mse_and_denormalizedMse
                Legend_nonparametric_dict[non_parametric_model_name_dataset] = non_parametric_model_name
                Result_mag_nonparametric_mean_dict[non_parametric_model_name_dataset] = denormalized_mse_mag
                Result_mag_nonparametric_min_dict[non_parametric_model_name_dataset] = denormalized_mse_mag
                Result_mag_nonparametric_max_dict[non_parametric_model_name_dataset] = denormalized_mse_mag
                Result_mag_nonparametric_std_dict[non_parametric_model_name_dataset] = 0
                Result_ang_nonparametric_mean_dict[non_parametric_model_name_dataset] = denormalized_mse_ang
                Result_ang_nonparametric_min_dict[non_parametric_model_name_dataset] = denormalized_mse_ang
                Result_ang_nonparametric_max_dict[non_parametric_model_name_dataset] = denormalized_mse_ang
                Result_ang_nonparametric_std_dict[non_parametric_model_name_dataset] = 0
                Result_nonparametric_mean_dict[non_parametric_model_name_dataset] = denormalized_mse
                Result_nonparametric_min_dict[non_parametric_model_name_dataset] = denormalized_mse
                Result_nonparametric_max_dict[non_parametric_model_name_dataset] = denormalized_mse
                Result_nonparametric_std_dict[non_parametric_model_name_dataset] = 0

            # All results are in the dictionary. Generate the plots (1 per batch size)
            if plot_is_ready_for_creation:
                ytitleStr = "MSE"
                file_postfix = "_mse"
                keys_sorted = sorted(Result_mean_dict.keys())
                keys_nonparametric_sorted = sorted(Result_nonparametric_mean_dict.keys())

                output_image_name = outfile_prefix + "_bs" + str(batch_size) + "_Nv" + str(Nv) + file_postfix
                y_mag_lst_of_lists, y_ang_lst_of_lists, y_lst_of_lists = [], [], []
                std_mag_lst_of_lists, std_ang_lst_of_lists, std_lst_of_lists = [], [], []
                legend_lst = []
                # stack the parametric plots (one plot line per modeltype-lambda-T value)
                list_of_wls_Ts_that_were_added_to_plot = []
                for model_type, lambda_, T in itertools.product(model_type_list, lambdas, Ts):

                    # WLS works with lambda 0 only
                    if model_type == 'wls':
                        if T in list_of_wls_Ts_that_were_added_to_plot:
                            continue
                        else:
                            lambda_ = int(0)
                            list_of_wls_Ts_that_were_added_to_plot.append(T)

                    y_mag_list, y_ang_list, y_list = [], [], []
                    std_mag_list, std_ang_list, std_list = [], [], []
                    # The following loop fills the values that correspond to the x axis
                    for Ns in Nss:
                        key = "{}_{}_{}_{}".format(model_type, lambda_, T, Ns)
                        y_mag_list.append(Result_mag_mean_dict[key])
                        std_mag_list.append(Result_mag_std_dict[key])
                        y_ang_list.append(Result_ang_mean_dict[key])
                        std_ang_list.append(Result_ang_std_dict[key])
                        y_list.append(Result_mean_dict[key])
                        std_list.append(Result_std_dict[key])
                    y_mag_lst_of_lists.append(y_mag_list)
                    std_mag_lst_of_lists.append(std_mag_list)
                    y_ang_lst_of_lists.append(y_ang_list)
                    std_ang_lst_of_lists.append(std_ang_list)
                    y_lst_of_lists.append(y_list)
                    std_lst_of_lists.append(std_list)
                    legend_key = "{}_{}_{}".format(model_type, lambda_, T)
                    legend_lst.append(Legend_dict[legend_key])
                # add the non-parametric plots.
                for non_parametric_model_name in non_parametric_model_name_list:
                    non_parametric_model_name_dataset = "{}_{}".format(non_parametric_model_name, dataset_name)
                    y_mag_lst_of_lists.append([Result_mag_nonparametric_mean_dict[non_parametric_model_name_dataset]]*len(Nss))
                    std_mag_lst_of_lists.append([Result_mag_nonparametric_std_dict[non_parametric_model_name_dataset]]*len(Nss))
                    y_ang_lst_of_lists.append([Result_ang_nonparametric_mean_dict[non_parametric_model_name_dataset]]*len(Nss))
                    std_ang_lst_of_lists.append([Result_ang_nonparametric_std_dict[non_parametric_model_name_dataset]]*len(Nss))
                    y_lst_of_lists.append([Result_nonparametric_mean_dict[non_parametric_model_name_dataset]]*len(Nss))
                    std_lst_of_lists.append([Result_nonparametric_std_dict[non_parametric_model_name_dataset]]*len(Nss))
                    legend_lst.append(Legend_nonparametric_dict[non_parametric_model_name_dataset])
                # Do the plotting already
                extras_dict = {"linewidth_list" : [2]*len(y_lst_of_lists),
                               "legend_location": (args.legend_x,args.legend_y) if args.legend_x!=-1 and args.legend_y != -1 else 0,
                               "y_axis_scale"   : 'log',
                               'font_size'      : 18,
                               "kill_errorbar"  : not args.keep_std_bars}
                plotManyLines(common_x_lst=Nss, y_lst_of_lists=y_mag_lst_of_lists, legends_lst=legend_lst,
                              xtitleStr=xtitleStr, ytitleStr=ytitleStr, titleStr=titleStr,
                              outputDir=args.output_directory, filename=output_image_name + "_mag",
                              extras_dict=extras_dict,
                              std_lst_of_lists=std_mag_lst_of_lists)
                plotManyLines(common_x_lst=Nss, y_lst_of_lists=y_ang_lst_of_lists, legends_lst=legend_lst,
                              xtitleStr=xtitleStr, ytitleStr=ytitleStr, titleStr=titleStr,
                              outputDir=args.output_directory, filename=output_image_name + "_ang",
                              extras_dict=extras_dict,
                              std_lst_of_lists=std_ang_lst_of_lists)
                plotManyLines(common_x_lst=Nss, y_lst_of_lists=y_lst_of_lists, legends_lst=legend_lst,
                              xtitleStr=xtitleStr, ytitleStr=ytitleStr, titleStr=titleStr,
                              outputDir=args.output_directory, filename=output_image_name,
                              extras_dict=extras_dict,
                              std_lst_of_lists=std_lst_of_lists)


                # Create a Pandas dataframe and store it to an excel sheet.
                Model_column = []
                T_column = []
                lambda_column = []
                Ns_column = []
                MSE_mag_column = []
                MSE_mag_stdcolumn = []
                MSE_ang_column = []
                MSE_ang_stdcolumn = []
                MSEcolumn = []
                MSEstdcolumn = []
                for model_type, lambda_, T, Ns in itertools.product(model_type_list, lambdas, Ts, Nss):
                    # WLS works with lambda 0 only
                    if model_type == 'wls':
                        if lambda_ != 0:
                            continue
                        else:
                            lambda_ = int(0)

                    for Ns in Nss:
                        key = "{}_{}_{}_{}".format(model_type, lambda_, T, Ns)
                        Model_column.append(model_type)
                        T_column.append(T)
                        lambda_column.append(lambda_)
                        Ns_column.append(Ns)
                        MSE_mag_column.append(Result_mag_mean_dict[key])
                        MSE_mag_stdcolumn.append(Result_mag_std_dict[key])
                        MSE_ang_column.append(Result_ang_mean_dict[key])
                        MSE_ang_stdcolumn.append(Result_ang_std_dict[key])
                        MSEcolumn.append(Result_mean_dict[key])
                        MSEstdcolumn.append(Result_std_dict[key])
                for non_parametric_model_name in non_parametric_model_name_list:
                    non_parametric_model_name_dataset = "{}_{}".format(non_parametric_model_name, dataset_name)
                    MSE_mag_column.append(Result_mag_nonparametric_mean_dict[non_parametric_model_name_dataset])
                    MSE_mag_stdcolumn.append(Result_mag_nonparametric_std_dict[non_parametric_model_name_dataset])
                    MSE_ang_column.append(Result_ang_nonparametric_mean_dict[non_parametric_model_name_dataset])
                    MSE_ang_stdcolumn.append(Result_ang_nonparametric_std_dict[non_parametric_model_name_dataset])
                    MSEcolumn.append(Result_nonparametric_mean_dict[non_parametric_model_name_dataset])
                    MSEstdcolumn.append(Result_nonparametric_std_dict[non_parametric_model_name_dataset])

                df = pd.DataFrame({'Model':    Model_column+non_parametric_model_name_list,
                                   'T': T_column + ["-"]*len(non_parametric_model_name_list),
                                   'lambda': lambda_column + ["-"]*len(non_parametric_model_name_list),
                                   'Ns': Ns_column + ["-"]*len(non_parametric_model_name_list),
                                   'denormalized MSE magnitude': MSE_mag_column,
                                   'denormalized MSE magnitude std':MSE_mag_stdcolumn,
                                   'denormalized MSE angle': MSE_ang_column,
                                   'denormalized MSE angle std': MSE_ang_stdcolumn,
                                   'denormalized MSE': MSEcolumn,
                                   'denormalized MSE std':MSEstdcolumn})

                # Create a Pandas Excel writer using XlsxWriter as the engine.
                append_df_to_excel(outfile_excel_path, df, sheet_name='Batch size {}'.format(batch_size), truncate_sheet=True)
                # with pd.ExcelWriter(outfile_excel_path) as writer:  # doctest: +SKIP
                #     df.to_excel(writer, sheet_name='Batch size {}'.format(batch_size), engine='xlsxwriter')
            else:
                ld("Skipping plot {} - not all the non parametric model logs were detected".format(outfile_prefix))
        else:
            ld("Skipping plot {} - not all the {} log files were detected".format(outfile_prefix, model_name_dataset))
