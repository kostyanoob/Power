import numpy as np
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.text import TextPath
matplotlib.use('Agg')
import matplotlib.pylab as pylab
import sys
import os
from matplotlib import colors as mcolors
from scipy.signal import savgol_filter
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)



def savefig(outputDir, filename, dpi=300, save_copy_as_eps=True, eps_dpi=1000, verbose=True):
    """
    Saves the current figure to an image file.

    :param outputDir: <str> a path to the directory where the new file will be created
    :param filename: <str> the name of the filename to be created. Can be specified with an extension
                           such as "image1.jpg" and then the format will be inferred. Otherwise, when
                           no extension is specified - c.f. "image1", the default PNG format is used.
    :param dpi: <int> (default 300) the dpi resolution of the image to be saved
    :param save_copy_as_eps: <bool> (default True) True iff an eps (vector-graphics) copy should be created
    * Removes all commas from the filename, except for the last comma - crucial for the LaTex *

    :param eps_dpi: <int> (default 1000) the dpi resolution of the eps image to be saved
    :param verbose: <bool> True iff messages regarding file saving success should be displayed.
    :return: nothing
    """
    # pylab.rcParams.update({'figure.autolayout': True})
    assert(len(filename)>0)
    filename_list = filename.split(".")
    if len(filename_list) >= 1 and not(filename_list[-1] in pylab.gcf().canvas.get_supported_filetypes().keys()):
        filename = filename + ".png"

    pylab.savefig(os.path.join(outputDir, filename), dpi=dpi, bbox_inches = "tight")
    if verbose:
        print("Saved plot to:" + os.path.join(outputDir, filename))
    if save_copy_as_eps:
        filename_list = filename.split(".")
        if len(filename_list) > 1 and filename_list[-1] in pylab.gcf().canvas.get_supported_filetypes().keys():
            filename_list[-1] = ".eps"
        elif len(filename_list) == 1:
            filename_list.append(".eps")
        else:
            raise Exception("Could not store the eps image: Illegal filename")
        filename_eps = "".join(filename_list)
        pylab.savefig(os.path.join(outputDir, filename_eps), format='eps', dpi=eps_dpi, bbox_inches = "tight")
        if verbose:
            print("Saved plot to:" + os.path.join(outputDir, filename_eps))

def plotOneLine(yAxisLst, xAxisLst, xtitleStr, ytitleStr, titleStr, outputDir, filename, stds=None,
                customPointAnnotation=None, verbose=False):
    '''
    Creates a single line plot and stores it to a file.
    : param stds - if is none, then the plot is a simple line, otherwise it
                   is interpreted as a list of standard deviations and added to the plot
    : param customPointAnnotation - set to a particular x value to mark the plot-line
                            with an arrow at this x value.
                            can be a list - if multiple annotated points are deisred
    '''

    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug
    fig = pylab.figure()
    pylab.xlabel(xtitleStr)
    pylab.ylabel(ytitleStr)
    pylab.suptitle(titleStr)
    if stds is None:
        pylab.plot(xAxisLst, yAxisLst)
    else:
        pylab.errorbar(xAxisLst, yAxisLst, stds, ecolor="#AAAAAA")
    axes = pylab.gca()
    axes.set_xlim([min(xAxisLst), max(xAxisLst)])

    if not(customPointAnnotation is None):
        if type(customPointAnnotation)!=list:
            customPointAnnotation = [customPointAnnotation]
        for pt in customPointAnnotation:
            if (pt >= min(xAxisLst) or pt <=max(xAxisLst)):
                annotation_mark_x = pt
                annotation_mark_y = min(yAxisLst)
                pylab.plot([annotation_mark_x], [annotation_mark_y], '^', markersize=5)
                pylab.annotate("", xy=(annotation_mark_x, annotation_mark_y),
                               arrowprops=dict(facecolor='orange', shrink=0.05))
            else:
                ld("Warning: cannot annotate plot at point {}, since it's out of the range of X-axis".format(pt))


    # axes.set_ylim([min(lossVector),max(lossVector)+0.1])
    if min(yAxisLst) != 0 and max(yAxisLst) / min(yAxisLst) > 1000:
        axes.set_yscale('log')
    try:
        # Save to file both to the required format and to png
        savefig(outputDir, filename, save_copy_as_eps=True, verbose=verbose)
    except:
        pass
    pylab.close(fig)


def plotTwoLines(trainAcc, validAcc, xAxisLst, xtitleStr, ytitleStr, titleStr, outputDir, filename, isAnnotatedMax=False, isAnnotatedMin=False,
                 trainStr='Train', validStr='Validation', customPointAnnotation=None):
    '''

    : param customPointAnnotation - set to a particular x value to mark both of the plot-lines
                                    with an arrow at this x value.
                                    can be a list - if multiple annotated points are deisred

    Legend "loc" arguments:
    'best'         : 0, (only implemented for axes legends)
    'upper right'  : 1,
    'upper left'   : 2, <--- we chose it
    'lower left'   : 3,
    'lower right'  : 4,
    'right'        : 5,
    'center left'  : 6,
    'center right' : 7,
    'lower center' : 8,
    'upper center' : 9,
    'center'       : 10,
    '''
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug
    fig = pylab.figure()
    pylab.xlabel(xtitleStr)
    pylab.ylabel(ytitleStr)
    pylab.suptitle(titleStr)
    plot_train, = pylab.plot(xAxisLst, trainAcc, label=trainStr, linestyle="--")
    plot_valid, = pylab.plot(xAxisLst, validAcc, label=validStr, linestyle="-")
    pylab.legend([plot_train, plot_valid], [trainStr, validStr], loc=0)
    if (isAnnotatedMax or isAnnotatedMin):
        annotationIdx = np.argmax(validAcc) if isAnnotatedMax else np.argmin(validAcc)
        annotationVal = validAcc[annotationIdx]
        minAcc_total = min(min(validAcc), min(trainAcc))
        maxAcc_total = max(max(validAcc), max(trainAcc))
        stry = validStr + " %.1f%%" % annotationVal + " at epoch " + str((annotationIdx + 1))
        pylab.plot([annotationIdx + 1], [annotationVal], 'o')
        pylab.annotate(stry, xy=(annotationIdx + 1, annotationVal),
                       xytext=(annotationIdx + 1 - len(xAxisLst) * 0.25, annotationVal - (maxAcc_total - minAcc_total) / 10),
                       arrowprops=dict(facecolor='orange', shrink=0.05))
    if not(customPointAnnotation is None):
        if type(customPointAnnotation)!=list:
            customPointAnnotation = [customPointAnnotation]
        for pt in customPointAnnotation:
            if (pt in xAxisLst):
                pylab.plot([pt, pt], [trainAcc[pt], validAcc[pt]], 'o', markersize=3)
                pylab.annotate("", xy=(pt, trainAcc[pt]),
                               arrowprops=dict(facecolor='orange', shrink=0.05))
                pylab.annotate("", xy=(pt, validAcc[pt]),
                               arrowprops=dict(facecolor='orange', shrink=0.05))
            else:
                ld("Warning: cannot annotate plot at point {}, since it's out of the range of X-axis".format(pt))
    # Save to file both to the required format and to png
    savefig(outputDir, filename, save_copy_as_eps=True)
    pylab.close(fig)


def plotBars(matrix_of_bars, list_of_labels, xtitleStr, ytitleStr, titleStr, outputDir, filename):
    '''
    Displays the average of each row and its std - as a bar.

    :param listOfbar_lists: a 2D array, each row representing a bar measurments
    :param list_of_labels: a list of strings,
           corresponding to the number of rows in the matrix_of_bars.
    :return:
    '''
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug
    badInput = True
    try:
        badInput = len(matrix_of_bars.shape) != 2
    except Exception:
        badInput = True
    if badInput:
        raise Exception("Error at plotBars: matrix_of_bars argument must be a 2D numpy array")
    n_groups = matrix_of_bars.shape[0]
    badInput = True
    try:
        badInput = len(list_of_labels) != n_groups
    except Exception:
        badInput = True
    if badInput:
        raise Exception("Error at plotBars: list_of_labels argument must contain enough labels for all the rows in the a matrix_of_bars")


    means = np.mean(matrix_of_bars, axis=1)
    stds = np.std(matrix_of_bars, axis=1)

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    fig = pylab.figure()
    rects1 = pylab.bar(index, means, bar_width,
                     alpha=opacity,
                     color='b',
                     yerr=stds,
                     error_kw=error_config)
                     #label='Accuracy')


    pylab.xlabel(xtitleStr)
    pylab.ylabel(ytitleStr)
    pylab.title(titleStr)
    pylab.xticks(index + bar_width / 2, list_of_labels, rotation = 'vertical')
    pylab.legend()

    pylab.tight_layout()

    # Save to file both to the required format and to png
    savefig(outputDir, filename, save_copy_as_eps=True)
    pylab.close(fig)


def plotManyLines(common_x_lst, y_lst_of_lists, legends_lst, xtitleStr, ytitleStr, titleStr,
                  outputDir, filename, extras_dict=None, customPointAnnotation=None,
                  std_lst_of_lists=None):
    '''


    for python 3.5
    :param common_x_lst:
    :param y_lst_of_lists:
    :param legends_lst:
    :param xtitleStr:
    :param ytitleStr:
    :param titleStr:
    :param outputDir:
    :param filename:
    :return:
    '''

    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug
    if 'font_size' in extras_dict:
        pylab.rcParams.update({'font.size': extras_dict['font_size']})
    fig = pylab.figure()
    pylab.xlabel(xtitleStr)
    pylab.ylabel(ytitleStr)
    pylab.suptitle(titleStr)
    colors = ['r','g','b','c','m','y','k',"#96f97b",'#ae7181','#0504aa', '#c1f80a', '#b9a281', '#ff474c'][:len(legends_lst)]
    if extras_dict is None:
        linewidth_list = [2]*len(y_lst_of_lists)
        legend_location = 0
        pylab.yscale('linear')
        marker_list = ["."] * len(y_lst_of_lists)
    else:
        linewidth_list = extras_dict["linewidth_list"]
        legend_location = extras_dict["legend_location"]
        if 'y_axis_scale' in extras_dict:
            pylab.yscale(extras_dict["y_axis_scale"])
        if 'marker_list' not in extras_dict:
            marker_list = ['s', 'x', '*', '^', 'p', 'D', '>', '<', '+','o','.'][:len(legends_lst)] if 'marker_list' not in extras_dict else extras_dict['extras_dict']
        else:
            marker_list = extras_dict['marker_list']


    if std_lst_of_lists is None or 'kill_errorbar' in extras_dict and extras_dict['kill_errorbar']:
        axes = [pylab.plot(common_x_lst, y_lst, label=legenda, linewidth=linewidth, color=cllr, marker=marker) for cllr,y_lst,marker,linewidth,legenda in zip(colors,y_lst_of_lists,marker_list,linewidth_list,legends_lst)]
    else:
        axes = [pylab.errorbar(common_x_lst, y_lst, std_, label=legenda, linewidth=linewidth, color=cllr, marker=marker, elinewidth=errorLineSize, capsize=errorLineSize)
                for cllr, y_lst, std_,marker, linewidth, legenda, errorLineSize in zip(colors, y_lst_of_lists, std_lst_of_lists, marker_list,linewidth_list, legends_lst, list(range(2,len(legends_lst)+2)))]
    #pylab.legend(handles = [mpatches.Patch(color =cllr, label=legenda) for cllr, legenda in zip(colors,legends_lst)])
    pylab.legend(loc=legend_location, markerscale=2)


    if not(customPointAnnotation is None):
        if type(customPointAnnotation)!=list:
            customPointAnnotation = [customPointAnnotation]
        for pt in customPointAnnotation:
            if (pt >= min(common_x_lst) or pt <=max(common_x_lst)):
                annotation_mark_x = pt
                annotation_mark_y = min([min(t) for t in y_lst_of_lists])
                pylab.plot([annotation_mark_x], [annotation_mark_y], '^', markersize=5)
                pylab.annotate("", xy=(annotation_mark_x, annotation_mark_y),
                               arrowprops=dict(facecolor='#AAAAAA', shrink=0.05))
            else:
                ld("Warning: cannot annotate plot at point {}, since it's out of the range of X-axis".format(pt))

    # Save to file both to the required format and to png
    savefig(outputDir, filename, save_copy_as_eps=True)
    pylab.close(fig)


def plotManyBars(common_x_lst, y_lst_of_lists, legends_lst, xtitleStr, ytitleStr, titleStr, outputDir, filename, customPointAnnotation=None, list_of_y_stds=None):
    '''
    Creates a bar plot with multiple
    '''
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug
    fig = pylab.figure()
    pylab.xlabel(xtitleStr)
    pylab.ylabel(ytitleStr)
    pylab.suptitle(titleStr)
    #colors = list(mcolors.CSS4_COLORS.values())[14:14 + len(legends_lst)]
    colors = ['r','g','b','c','m','y','k'][:len(legends_lst)]
    if type(common_x_lst) == list:
        common_x_lst = np.array(common_x_lst)
    elif type(common_x_lst) != np.ndarray:
        raise Exception("Bad common_x_lst. Must be either list of numpy 1D array.")

    nBarsPerFamily = len(common_x_lst)
    nFamilies = len(y_lst_of_lists)
    nTotalBars = nFamilies * nBarsPerFamily
    span = common_x_lst.max() - common_x_lst.min()
    bar_width_including_delimiters = span / (nTotalBars)
    cluster_width = nFamilies * bar_width_including_delimiters
    delimiter_width = 0.1 * cluster_width
    bar_width = bar_width_including_delimiters-delimiter_width/nFamilies
    families_X_offset = cluster_width*0.9/2
    ld("BPF:{} nFamilies:{} nTotalBars:{} span:{} bar_width:{}".format(nBarsPerFamily,nFamilies,nTotalBars,span,bar_width))
    opacity = 0.7

    if not list_of_y_stds is None:
        error_config = {'ecolor': '0.3'}
        axes = [pylab.bar(common_x_lst - families_X_offset + i*(delimiter_width*int(i==0)+bar_width), y_lst,   bar_width,
                          alpha=opacity, color=cllr,
                          yerr=std_t, error_kw=error_config,
                          label=lgnd) for cllr, y_lst, lgnd, std_t,i in zip(colors, y_lst_of_lists, legends_lst, list_of_y_stds,range(len(y_lst_of_lists)))]
    else:
        axes = [pylab.bar(common_x_lst - families_X_offset +cluster_width/2 + i*(delimiter_width*int(i==0)+bar_width), y_lst, bar_width,
                          alpha=opacity, color=cllr,
                          label=lgnd) for cllr, y_lst, lgnd,i in zip(colors, y_lst_of_lists, legends_lst,range(len(y_lst_of_lists)))]
    pylab.legend(handles=[mpatches.Patch(color=cllr, label=legenda, alpha=opacity) for cllr, legenda in zip(colors, legends_lst)])

    if not(customPointAnnotation is None):
        if type(customPointAnnotation)!=list:
            customPointAnnotation = [customPointAnnotation]
        for pt in customPointAnnotation:
            if (pt >= min(common_x_lst) or pt <=max(common_x_lst)):
                annotation_mark_x = pt
                annotation_mark_y = min([min(t) for t in y_lst_of_lists])
                pylab.plot([annotation_mark_x], [annotation_mark_y], '^', markersize=5)
                pylab.annotate("", xy=(annotation_mark_x, annotation_mark_y),
                               arrowprops=dict(facecolor='#AAAAAA', shrink=0.05))
            else:
                ld("Warning: cannot annotate plot at point {}, since it's out of the range of X-axis".format(pt))

    # Save to file both to the required format and to png
    savefig(outputDir, filename, save_copy_as_eps=True)
    pylab.close(fig)


def plotListOfPlots(x_lst_of_lists, y_lst_of_lists, legends_lst, xtitleStr, ytitleStr, titleStr, outputDir, filename, lpf=None, colorLst=None, fontsize=None, showGrid=False):
    '''
        :param lpf: the window-length of averaging. This is used for smoothing, and implemented by the Savitzky-Golay
                    filter.
    '''
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug

    fig = pylab.figure()
    pylab.xlabel(xtitleStr, fontsize=fontsize)
    pylab.ylabel(ytitleStr, fontsize=fontsize)

    if not titleStr is None and titleStr != "":
        pylab.suptitle(titleStr)
    #colors = list(mcolors.CSS4_COLORS.values())
    if colorLst is None:
        colorLst = ['red', 'orange', 'green', 'blue', 'darkblue', 'purple', 'black', 'yellow']

    if lpf != None:
        y_lst_of_lists_new =[savgol_filter(np.array(data), lpf, 1) for data in y_lst_of_lists]
        y_lst_of_lists = y_lst_of_lists_new

    if not fontsize is None:
        #matplotlib.rcParams.update({'font.size': fontsize})
        ##matplotlib.rc('xtick', labelsize=fontsize)
        #matplotlib.rc('ytick', labelsize=fontsize)
        # pylab.rc('font', size=fontsize)  # controls default text sizes
        # pylab.rc('axes', titlesize=fontsize)  # fontsize of the axes title
        # pylab.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
        # pylab.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
        # pylab.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels
        # pylab.rc('legend', fontsize=fontsize)  # legend fontsize
        # pylab.rc('figure', titlesize=fontsize)  # fontsize of the figure title
        pass

    axes = [pylab.plot(x_lst, y_lst, color=cllr, linewidth=3.0) for cllr, x_lst,y_lst in zip(colorLst, x_lst_of_lists, y_lst_of_lists)]
    if not legends_lst is None and len(legends_lst) == len(x_lst_of_lists):
        pylab.legend(handles = [mpatches.Patch(color =cllr, label=legenda) for cllr, legenda in zip(colorLst, legends_lst)])
    #pylab.legend(axes, legends_lst, loc=0)    # old legend generation
    if showGrid:
        pylab.gca().grid(True, which='both', linestyle=':')
    # Save to file both to the required format and to png
    savefig(outputDir, filename, save_copy_as_eps=True)
    pylab.close(fig)


def plotListOfScatters(x_lst_of_lists, y_lst_of_lists, legends_lst, xtitleStr, ytitleStr, titleStr, outputDir, filename, extras_dict={}):
    """

    :param x_lst_of_lists:
    :param y_lst_of_lists:
    :param legends_lst:
    :param xtitleStr:
    :param ytitleStr:
    :param titleStr:
    :param outputDir:
    :param filename:
    :param extras_dict: can contain weird options such as
        'legend_location' : 0,1,2,3,4... - <int> sets the location to be other than just "0", which is the "best" automatic
        'marker_list' : <list> which must be of the size of the number of different plots in the figure. Describing a  marker that will be applied for each plot.
    :return:
    """
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug

    fig = pylab.figure()
    pylab.xlabel(xtitleStr)
    pylab.ylabel(ytitleStr)
    if titleStr != "":
        pylab.suptitle(titleStr)

    """
    #colors = list(mcolors.CSS4_COLORS.values())[14:14 + len(legends_lst)]
    markersize=10
    subsampleFactor=4
    colors = ['g','b','c','r','m','y','k',"#96f97b",'#ae7181','#0504aa', '#c1f80a', '#b9a281', '#ff474c'][:len(legends_lst)]
    markerLst = ['s','x','+','*', '^', 'p', 'D', '>', '<']
    axes = [pylab.plot(common_x_lst[::subsampleFactor], y_lst[::subsampleFactor], color=cllr, marker=mrkr, markersize=markersize) for cllr,mrkr,y_lst in zip(colors,markerLst,y_lst_of_lists)]
    pylab.legend(handles = [mlines.Line2D([0], [0], color=cllr, marker=mrkr, lw=3.0, markersize=markersize+2, label=legenda) for cllr,mrkr,legenda in zip(colors,markerLst,legends_lst)])

    """

    # extract extra fancy features for the plot

    legend_location = 0 if 'legend_location' not in extras_dict else extras_dict['legend_location']
    marker_list = ["."]*len(x_lst_of_lists) if 'marker_list' not in extras_dict else extras_dict['marker_list']
    marker_scaler_list = [2] * len(x_lst_of_lists) if 'marker_scaler_list' not in extras_dict else extras_dict['marker_scaler_list']
    # import pdb
    # pdb.set_trace()
    colorVocabulary = ['red', 'orange', 'green', 'lightgreen', 'darkblue', 'cyan', 'purple', 'pink', 'black', 'gray', 'brown', 'darkred']
    colorLst = [(cllr,'none') if i%2==1 else ('none',cllr) for (i,cllr) in enumerate(colorVocabulary)]
    if 'font_size' in extras_dict:
        matplotlib.rcParams.update({'font.size': extras_dict['font_size']})
        matplotlib.rcParams['legend.fontsize'] = extras_dict['font_size']
    # create the actual plots
    axes = [pylab.scatter(x_lst, y_lst, s=marker_size, facecolors=cllr[0], edgecolors=cllr[1], marker=marker) for x_lst,y_lst,cllr,marker,marker_size in zip(x_lst_of_lists, y_lst_of_lists,colorLst,marker_list,marker_scaler_list)]
    pylab.legend(axes, legends_lst, loc=legend_location, markerscale=2)

    # Save to file both to the required format and to png
    savefig(outputDir, filename, save_copy_as_eps=True, verbose=False)
    pylab.close(fig)


def plotBetweens(y_list, xAxisLst, xtitleStr, ytitleStr, titleStr, legendStrLst, outputDir, filename, verbose=True, hsv_colors=False):
    # type: (list, list, str, str, str, list, str, str, bool, bool) -> None
    '''
    The y_list constains multiple y-values, all consistent with the xAxisLst ticks
    The plot will consist of len(Y_list) plots, with a different color fill between two consecutive plots.

    Pre-conditions:
    1) The first array (or list) in y_list is assumed to be the lowest one in its height.
    2) The first array (or list) in y_list will be filled downwards till the x-axis.

    y
    ^                                 _________ y_list[2]
    |                                /
    |       color=white             /  color2
    |                              /
    |                       ______/___________ y_list[1]
    |                      /
    |_____________________/     color1
    |                   ______________________ y_list[0]
    |  color1   _______/
    |          /              color0
    |_________/
    ------------------------------------------->x

    Legend "loc" arguments:
    'best'         : 0, (only implemented for axes legends)
    'upper right'  : 1,
    'upper left'   : 2, <--- we chose it
    'lower left'   : 3,
    'lower right'  : 4,
    'right'        : 5,
    'center left'  : 6,
    'center right' : 7,
    'lower center' : 8,
    'upper center' : 9,
    'center'       : 10,
    '''
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
        ld = logging.debug

    fig = pylab.figure(figsize=(8,2.25))
    pylab.xlabel(xtitleStr)
    pylab.ylabel(ytitleStr)
    pylab.suptitle(titleStr)

    # Add the artificial "zero-plot" to be the reference plot
    zerolst = len(xAxisLst) * [0]
    y_arr_with_zerolst = [zerolst] + y_list
    axis_lst = [pylab.plot(xAxisLst, zerolst, "k-")]
    legend_handles = []

    # Color list for assining different colors to different fills
    # from __future__ import division
    #
    # import matplotlib.pyplot as plt
    # from matplotlib import colors as mcolors
    if hsv_colors:
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

        # Sort colors by hue, saturation, value and name.
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                        for name, color in colors.items())
        color_lst = [name for hsv, name in by_hsv]
    else:
        color_lst = ['tab:black', 'tab:orange', 'tab:red', 'tab:brown', 'tab:purple', 'tab:pink', 'tab:olive',
                     'tab:green','tab:pink', 'tab:olive', 'tab:green','tab:gray', 'tab:cyan', 'tab:blue']
    num_plots = len(y_arr_with_zerolst)
    num_colors = len(color_lst)

    # Add the plot-lines and fill the space between each to consequitive ones
    for i in range(1, num_plots):
        fill_color = color_lst[i % min(num_plots, num_colors)]
        axis_lst.append(pylab.plot(xAxisLst, y_arr_with_zerolst[i], "k-", linewidth=0.2))
        pylab.fill_between(xAxisLst, y_arr_with_zerolst[i], y_arr_with_zerolst[i - 1], facecolor=fill_color,
                           interpolate=True)
        legend_handles.append(mpatches.Patch(color=fill_color, label=legendStrLst[i - 1]))

    # Legends

    pylab.legend(handles=legend_handles[::-1])#, loc=2)

    # Save to file both to the required format and to png
    savefig(outputDir, filename, save_copy_as_eps=True, verbose=verbose)
    pylab.close(fig)

    #subsample the bar data


def subsample(x,y,factor):
    new_x = []
    new_y = []
    for i in range(0,len(x),factor):
        new_x.append(x[i])
        new_y.append(0)
        for j in range(factor):
            new_y[-1] += y[i+j]

    return new_x, new_y


def plotListOfPlots_and_Bars(x_lst_of_lists, y_lst_of_lists, legends_lst, xtitleStr, ytitleStr, titleStr, outputDir, filename, n_lines, lpf=None, colorLst=None, fontsize=None, showGrid=False):
    '''
    : param nlines indicates how many lines there are in the list. after this amount of lines - the bars begin
        :param lpf: the window-length of averaging. This is used for smoothing, and implemented by the Savitzky-Golay
                    filter.
    '''
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug

    fig = pylab.figure()
    pylab.xlabel(xtitleStr, fontsize=fontsize)
    pylab.ylabel(ytitleStr, fontsize=fontsize)

    if not titleStr is None and titleStr != "":
        pylab.suptitle(titleStr)
    #colors = list(mcolors.CSS4_COLORS.values())
    if colorLst is None:
        colorLst = ['red', 'orange', 'green', 'blue', 'darkblue', 'purple', 'black']

    if lpf != None:
        y_lst_of_lists_new =[savgol_filter(np.array(data), lpf, 1) for data in y_lst_of_lists]
        y_lst_of_lists = y_lst_of_lists_new

    if not fontsize is None:
        #matplotlib.rcParams.update({'font.size': fontsize})
        ##matplotlib.rc('xtick', labelsize=fontsize)
        #matplotlib.rc('ytick', labelsize=fontsize)
        # pylab.rc('font', size=fontsize)  # controls default text sizes
        # pylab.rc('axes', titlesize=fontsize)  # fontsize of the axes title
        # pylab.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
        # pylab.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
        # pylab.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels
        # pylab.rc('legend', fontsize=fontsize)  # legend fontsize
        # pylab.rc('figure', titlesize=fontsize)  # fontsize of the figure title
        pass

    axes = []
    n_bars = len(y_lst_of_lists) - n_lines
    subsample_bar_factor = 10 # by how much is the bar plot's resolution is lower than the reliable accuracy's
    n_bins = len(y_lst_of_lists[0]) / subsample_bar_factor
    ld("Using " + str(n_bins) + " bins and " + str(n_bars) + "bars")

    bar_width = (max(x_lst_of_lists[0])-min(x_lst_of_lists[0])) / float(n_bars*n_bins)
    bin_offset = bar_width*(n_bars/2)


    for cllr, x_lst, y_lst,i in zip(colorLst, x_lst_of_lists, y_lst_of_lists, range(n_bars+n_lines)):
        if i < n_lines:
            axes.append(pylab.plot(x_lst, y_lst, color=cllr, linewidth=3.0))
        else:
            new_x, new_y = subsample(x_lst, y_lst, subsample_bar_factor)
            axes.append(pylab.bar(x_lst-bin_offset+(i-n_lines)*bar_width, y_lst, color=cllr, align='center', width=bar_width))


    if not legends_lst is None and len(legends_lst) == len(x_lst_of_lists):
        pylab.legend(handles = [mpatches.Patch(color =cllr, label=legenda) for cllr, legenda in zip(colorLst, legends_lst)])
    #pylab.legend(axes, legends_lst, loc=0)    # old legend generation
    if showGrid:
        pylab.gca().grid(True, which='both', linestyle=':')
    # Save to file both to the required format and to png
    savefig(outputDir, filename, save_copy_as_eps=True)
    pylab.close(fig)