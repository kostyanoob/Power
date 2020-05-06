"""
A script that generates multiple plots from log files of
the series_voltage runs (produced by the model_run_script_dfm.py)
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
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug

from Utils.logprints import is_log_complete

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


def logfile_MSEs(dir, log_filename):
    """
    Reads a logfile and returns the two MSE values written at the last line thereof.
    :param dir: a directory where all the log-files reside
    :param log_filename: the filename of the log
    :return: a 2-tuple corresponding to MSE and the denormalized-MSE
    """
    if not is_log_complete(dir, log_filename):
        return None

    with open(os.path.join(dir, log_filename),"r") as txtFile:
        log_last_line = txtFile.readlines()[-1]
        # Test MSE: normalized: 0.00001418512374983615 denormalized 0.00004358236030056035
        split_line = log_last_line.split(" ")
        mse = float(split_line[-3])
        denormalized_mse = float(split_line[-1])
    return mse, denormalized_mse


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
                    default='series_voltage',
                    help='Comma separated names of the model names, that should be aggregated.')
parser.add_argument('-non-parametric-model-name-list', type=str, metavar='<str>',
                    default='persistent',
                    help='Comma separated names of the model names, which were run without any parameters.')
parser.add_argument('-datasets', type=str, metavar='<str>',
                    default="solar_smooth_ord_60_downsampling_factor_60",
                    help='Datsets, comma separated strings')
parser.add_argument('-batch-sizes', type=str, metavar='<str>',
                    default="50,100,200",
                    help='batch sizes with which were performed, comma separated integers')
parser.add_argument('-seeds', type=str,
                    default="1,2,3,4,5,6,7,8", metavar='<str>',
                    help='random seeds with which the runs were performed, comma separated integers')
parser.add_argument('-Ts', type=str,
                    default="2,5,10,20,50,100",
                    help='Comma separated list of numbers of time steps in the inputs sliding window.')
parser.add_argument('-lambdas', type=str,
                    default="0.0,0.125,0.25,0.5,1.0,2.0,4.0,8.0,16.0",
                    help='Comma separated list of coefficients applied to the Power-Flow equation loss component.')
parser.add_argument('-plot_group_list', type=str, metavar='<str>',
                    default="lambda_T",
                    help='Comma separated names of plots to be generated.')

args = parser.parse_args()

model_name_list = args.model_name_list.split(",")
non_parametric_model_name_list = args.non_parametric_model_name_list.split(",")
plot_group_list = args.plot_group_list.split(",")
datasets = args.datasets.split(",")
batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
Ts = [int(x) for x in args.Ts.split(",")]
lambdas = [float(x) for x in args.lambdas.split(",")]
seeds = [int(x) for x in args.seeds.split(",")]

if not os.path.exists(args.output_directory):
    os.makedirs(args.output_directory)

titleStr = ""#"Sparsity in SimpleNet weights during training"
xtitleStr = "T"

###############################################################
################     Plot of lambda_T      ####################
###############################################################
outfile_prefix = "lambda_T"
outfile_excel_path = os.path.join(args.output_directory, outfile_prefix + ".xlsx")

if outfile_prefix in plot_group_list:
    ld("Running {} result aggregation!".format(outfile_prefix))


    for model_name, dataset_name, batch_size in list(itertools.product(model_name_list, datasets, batch_sizes)):

        model_name_dataset = "{}_{}".format(model_name, dataset_name)
        nesterov = "False"
        Result_mean_dict = {}
        Result_min_dict = {}
        Result_max_dict = {}
        Result_std_dict = {}
        Legend_dict = {}
        plot_is_ready_for_creation = True
        for lambda_, T in list(itertools.product(lambdas, Ts)):
            concatenated_parameters = "{}_{}_{}".format(batch_size, lambda_, T)
            Legend_dict[lambda_] = r"$\lambda={}$".format(lambda_)

            # Accumulate the current configuration with all the differently sed runs
            for seedid, seed in enumerate(seeds):

                log_filename = "{}_seed:{}_bs:{}_lambda:{}_T:{}.txt".format(model_name_dataset, seed, batch_size, lambda_, T)
                mse_and_denormalizedMse = logfile_MSEs(args.input_data_folder, log_filename)

                if mse_and_denormalizedMse is None:
                    plot_is_ready_for_creation = False
                    break
                else:
                    _, denormalized_mse = mse_and_denormalizedMse

                if not concatenated_parameters in Result_mean_dict:
                    Result_std_dict[concatenated_parameters] = np.zeros(len(seeds), dtype=np.float64)
                    Result_std_dict[concatenated_parameters][seedid] = denormalized_mse
                    Result_mean_dict[concatenated_parameters] = denormalized_mse
                    Result_min_dict[concatenated_parameters] = denormalized_mse
                    Result_max_dict[concatenated_parameters] = denormalized_mse
                else:
                    Result_std_dict[concatenated_parameters][seedid] = denormalized_mse
                    Result_min_dict[concatenated_parameters] = min(Result_min_dict[concatenated_parameters], denormalized_mse)
                    Result_max_dict[concatenated_parameters] = np.maximum(Result_max_dict[concatenated_parameters], denormalized_mse)
                    Result_mean_dict[concatenated_parameters] += denormalized_mse

            # Average out the seed runs
            if plot_is_ready_for_creation:
                Result_mean_dict[concatenated_parameters] /= len(seeds)
                Result_std_dict[concatenated_parameters] = np.std(Result_std_dict[concatenated_parameters], axis=0)
            else:
                break

        # Check out all the non-parametric models
        if plot_is_ready_for_creation:
            Result_nonparametric_mean_dict = {}
            Result_nonparametric_min_dict = {}
            Result_nonparametric_max_dict = {}
            Result_nonparametric_std_dict = {}
            Legend_nonparametric_dict = {}
            for non_parametric_model_name in non_parametric_model_name_list:
                non_parametric_model_name_dataset = "{}_{}".format(non_parametric_model_name, dataset_name)
                log_filename = "{}.txt".format(non_parametric_model_name_dataset)
                mse_and_denormalizedMse = logfile_MSEs(args.input_data_folder, log_filename)
                if mse_and_denormalizedMse is None:
                    plot_is_ready_for_creation = False
                    break
                else:
                    _, denormalized_mse = mse_and_denormalizedMse
                Legend_nonparametric_dict[non_parametric_model_name_dataset] = non_parametric_model_name
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

                output_image_name = outfile_prefix + "_batch_size_" + str(batch_size) + "_" + file_postfix
                y_lst_of_lists = []
                std_lst_of_lists = []
                legend_lst = []
                # stack the parametric plots (one plot line per lambda value)
                for lambda_ in lambdas:
                    y_list = []
                    std_list = []
                    for T in Ts:
                        key = "{}_{}_{}".format(batch_size, lambda_, T)
                        y_list.append(Result_mean_dict[key])
                        std_list.append(Result_std_dict[key])
                    y_lst_of_lists.append(y_list)
                    std_lst_of_lists.append(std_list)
                    legend_lst.append(Legend_dict[lambda_])
                # add the non-parametric plots.
                for non_parametric_model_name in non_parametric_model_name_list:
                    non_parametric_model_name_dataset = "{}_{}".format(non_parametric_model_name, dataset_name)
                    y_lst_of_lists.append([Result_nonparametric_mean_dict[non_parametric_model_name_dataset]]*len(Ts))
                    std_lst_of_lists.append([Result_nonparametric_std_dict[non_parametric_model_name_dataset]]*len(Ts))
                    legend_lst.append(Legend_nonparametric_dict[non_parametric_model_name_dataset])
                # Do the plotting already
                extras_dict = {"linewidth_list" : [2]*len(y_lst_of_lists),
                               "legend_location": 0,
                               "y_axis_scale"   : 'log',
                               "kill_errorbar"  : True}
                plotManyLines(common_x_lst=Ts, y_lst_of_lists=y_lst_of_lists, legends_lst=legend_lst,
                              xtitleStr=xtitleStr, ytitleStr=ytitleStr, titleStr=titleStr,
                              outputDir=args.output_directory, filename=output_image_name,
                              extras_dict=extras_dict,
                              std_lst_of_lists=std_lst_of_lists)


                # Create a Pandas dataframe and store it to an excel sheet.
                MSEcolumn = []
                MSEstdcolumn = []
                for lambda_ in lambdas:
                    for T in Ts:
                        key = "{}_{}_{}".format(batch_size, lambda_, T)
                        MSEcolumn.append(Result_mean_dict[key])
                        MSEstdcolumn.append(Result_std_dict[key])
                for non_parametric_model_name in non_parametric_model_name_list:
                    non_parametric_model_name_dataset = "{}_{}".format(non_parametric_model_name, dataset_name)
                    MSEcolumn.append(Result_nonparametric_mean_dict[non_parametric_model_name_dataset])
                    MSEstdcolumn.append(Result_nonparametric_std_dict[non_parametric_model_name_dataset])
                df = pd.DataFrame({'Model':    [model_name]*len(Ts)*len(lambdas)+non_parametric_model_name_list,
                                   'lambda': list(np.repeat(lambdas, len(Ts))) + ["-"]*len(non_parametric_model_name_list),
                                   'T': Ts*len(lambdas) + ["-"]*len(non_parametric_model_name_list),
                                   'denormalized MSE': MSEcolumn,
                                   'denormalized MSE std':MSEstdcolumn})

                # Create a Pandas Excel writer using XlsxWriter as the engine.
                append_df_to_excel(outfile_excel_path, df, sheet_name='Batch size {}'.format(batch_size), truncate_sheet=True)

            else:
                ld("Skipping plot {} - not all the non parametric model logs were detected".format(outfile_prefix))
        else:
            ld("Skipping plot {} - not all the {} log files were detected".format(outfile_prefix, model_name_dataset))
