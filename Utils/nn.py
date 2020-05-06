import tensorflow as tf
from math import sqrt
import logging

def countParams(exclude_from_counting=[]):
    """
    Returns the number of trainable parameters in the current graph
    :param exclude_from_counting: list of partial names such that if a
                                  trainable parameter contains any of the
                                  list contents, it will be excluded from
                                  parameter counting.
    :return: (int) number of parameters
    """
    total_parameters = 0
    for variable in tf.trainable_variables():

        if any(substr_grad_name in variable.name for substr_grad_name in exclude_from_counting):
            continue

        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters

    return total_parameters



def count_param():
    for variable in tf.trainable_variables():
        total_parameters = 0
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
        return total_parameters


# Create some wrappers for simplicity
def conv2d_relu(x, W, b, strides=1, pre_pad=0, use_batchnorm=False, batchnorm_MA_frac=0.95):
    '''
    Conv2D wrapper.
    1) Applies zero padding in the width and height dimensions (dim 1 and 2)
    2) Applies 2d-Convolution on the padded input with bias and relu activation
    '''
    pre_pad = int(pre_pad)
    strides = int(strides)
    x = tf.pad(x, [[0,0],[pre_pad,pre_pad],[pre_pad,pre_pad],[0,0]], "CONSTANT") # zero padding
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    if use_batchnorm:
        x = tf.contrib.layers.batch_norm(x, batchnorm_MA_frac)
    return tf.nn.relu(x)


def maxpool2d(x, k=2, strides=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding='VALID')

def avgpool2d(x, k=2, strides=2):
    # MaxPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding='VALID')

def CONV2dVar(ker_w, ker_h, in_channels, out_channels, name):
    '''
    Constructor of a weight matrix for a 2d-convolutional layer
    :param ker_w: convolutional kernel width
    :param ker_h: convolutional kernel width
    :param in_channels: convolutional kernel 3rd dimension (input channels number)
    :param out_channels: number of convolutional kernels
    :param name: the name of the variable, to be assigned
    :return: the tf.Variable, , defined for an intialization from ~N(0,sqrt(2.0/num_inputs_to_neuron))
    '''
    return tf.Variable(tf.random_normal([ker_w, ker_h, in_channels, out_channels], 0, sqrt(2.0 / (in_channels * ker_w * ker_h))), name=name)


def FCVar(n_inputs, n_outputs, name):
    '''
    Constructor of a weight matrix for a fully-connected layer
    :param n_inputs:
    :param n_outputs:
    :param name:
    :return: a tf.Variable, defined for an intialization from ~N(0,sqrt(2.0/num_inputs_to_neuron))
    '''
    return tf.Variable(tf.random_normal([n_inputs, n_outputs], 0,sqrt(2.0/n_inputs)), name=name)


def BiasVar(n_biases, name):
    '''
    Constructor of a bias vector
    :param n_biases:
    :param name:
    :return: a tf.Variable, defined for an intialization to zeros
    '''
    return tf.Variable(tf.zeros([n_biases]), name=name)


def resnet_module(input,
                  weights,
                  biases,
                  module_id,
                  n_resnet_blocks,
                  n_layers_in_block,
                  filter_sizes_in_block_lst,
                  subsample_on_first_block=False,
                  use_batchnorm = True,
                  batchnorm_MA_frac = 0.95):
    '''

    Constructs a ResNet module consisting of "n_resnet_blocks" blocks. Each block
    contains "n_layers_in_block" layers with the skip-connection going from the forst to last.
    Note that following the investigation of the ResNet block order of layers - it was found that the
    original paper implied the following order of layers in a ResNet block (example for n_layers_in_block=2):

    Input --> CONV --> BN --> ReLU ----> CONV --> BN --> ADD --> ReLU
       \________________________________________________/^

    :param input: the input tensor
    :param weights: dictionary with tf.Variables. Each variable should correspond to a single convolutional layer
                    parameters set. The key leading to this variable will be of the format "wc<module_id>.<block_id>.<layer_id>"
    :param biases: dictionary with tf.Variables. Each variable should correspond to a convolutional layer biases
                    .i.e to be a 1D array of a size equal to the number of filters of that very conv-layer. The name
                    format of the key is "bc<module_id>.<block_id>.<layer_id>"
    :param module_id: an integer used for identification of the proper weights and biases by their names.
    :param n_resnet_blocks: number of residual blocks to be cascaded
    :param n_layers_in_block : 2 for classical resnet block,
    :param filter_sizes_in_block_lst: a list with the filter numbers of every layer inside a block
                                   for example, if the bottleneck structure is desired, then
                                   the classical resnet block consists of two convolutional
                                   layers with the corresponding numbers of filters: [64,64,256]
    :param subsample_on_first_block: if True, then a subsampling of width and heigth
                                     of the input by will be performed by 2.
                                     The subsamplling is by the means of the convolutional
                                     striding in the first block out of the n_resnet_blocks

    :return:
    '''

    # construct first layer in the module (it may conatin a downsampling
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug



    ############# Layer 0 ######################################################
    # Optional sub-sampling in the first layer
    downsample_factor = 2
    strides = downsample_factor if subsample_on_first_block else 1
    pre_pad = filter_sizes_in_block_lst[0] / 2

    if subsample_on_first_block:

        # Step 1 - downsample the width and the height of the x
        x = avgpool2d(input, k=downsample_factor, strides=downsample_factor)
        #ld("x wxh was downsampled. x is now of a shape: {}".format(x.shape))

        # Step 2 - extend the 3rd dimension of the x's volume:
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, x.shape[-1]]], "CONSTANT")  # zero padding
        #ld("x num-channels was extended. x is now of a shape: {}".format(x.shape))

    else:

        x = input

    weightName = "wc{}.{}.{}".format(module_id, 0, 0)
    biasName = "bc{}.{}.{}".format(module_id, 0, 0)
    output = conv2d_relu(input, weights[weightName], biases[biasName], strides=strides, pre_pad=pre_pad, use_batchnorm=use_batchnorm, batchnorm_MA_frac=batchnorm_MA_frac)


    ###### All ther rest of n_resnet_blocks*n_layers_in_block - 1) layers #######
    for block in range(n_resnet_blocks):
        for layer in range(n_layers_in_block):

            if block==0 and layer==0:
                continue # first layer was already implemented outside these loops

            pre_pad = int(filter_sizes_in_block_lst[layer]/2)
            weightName = "wc{}.{}.{}".format(module_id, block, layer)
            biasName = "bc{}.{}.{}".format(module_id, block, layer)

            if layer == 0:
                x = output # prepare the shortcut origin

            if layer == n_layers_in_block-1: # apply the shortcut connection

                output = tf.pad(output, [[0, 0], [pre_pad, pre_pad], [pre_pad, pre_pad], [0, 0]], "CONSTANT")  # zero padding
                output = tf.nn.conv2d(output, weights[weightName], strides=[1, 1, 1, 1], padding='VALID')
                output = tf.nn.bias_add(output, biases[biasName])
                if use_batchnorm:
                    output = tf.contrib.layers.batch_norm(output, batchnorm_MA_frac)
                output = tf.add(output,x) # the holy grail of the resnet!
                output = tf.nn.relu(output)
            else:
                output = conv2d_relu(output, weights[weightName], biases[biasName], strides=1, pre_pad=pre_pad,
                                     use_batchnorm=use_batchnorm, batchnorm_MA_frac=batchnorm_MA_frac)

            #ld(" output shape is {}".format(output.shape))
    return output

def aggregate_batch_gradients(list_of_per_example_gradient_lists, clip_grads=None, use_median_gradients=True):
    '''
    Given a list with N lists of V gradients each, this procedure computes gradient-wise
    aggregation of the batch and outputs a list of aggregated gradients.

    if "clip_grads" is supplied, then the aggregated gradients are clipped to
    [-clip_grads,clip_grads] range prior to being aggregated.

    if "use_median_gradients=True" then median is used to aggregate the gradient of each variable
    if "use_median_gradients=False" then average is used to aggregate the gradient of each variable

    :return: a list containing V gradients.
    '''

    num_examples = len(list_of_per_example_gradient_lists)
    if len(list_of_per_example_gradient_lists)==0:
        return []

    num_gradients_tensors = len(list_of_per_example_gradient_lists[0])
    median_gradients = []

    for i in range(num_gradients_tensors):

        # if clip_grads is None:
        batch_of_current_var_gradients = tf.stack([list_of_per_example_gradient_lists[j][i] for j in range(num_examples)])
        # else:
        #     batch_of_current_var_gradients = tf.stack(
        #         [tf.clip_by_value(list_of_per_example_gradient_lists[j][i], -clip_grads, clip_grads) for j in range(num_examples)])
        #batch_of_current_var_gradients = tf.Print(batch_of_current_var_gradients, [tf.shape(batch_of_current_var_gradients)], "The shape of jungle jam {}".format(i))

        if not use_median_gradients:
            median_gradients.append(tf.reduce_mean(batch_of_current_var_gradients, axis=0))
        else:
            median_gradients.append(tf.contrib.distributions.percentile(batch_of_current_var_gradients, q=50.0, axis=0))
        #median_gradients.append(tf.reduce_mean(batch_of_current_var_gradients, axis=0))

    if not clip_grads is None:
        return [tf.clip_by_value(grad, -clip_grads, clip_grads) for grad in median_gradients]
    else:
        return median_gradients

def resnet(input_data, dropout_prob, weights, biases, use_batchnorm, batch_norm_MA_decay, resnet_n, resnet_n_layers_per_block, convPaddings, convFiltSizes, convStrides, averagePoolingFactor):
    '''


    :param input_data: input 4D Tensor containing the batch of input images
    :return:HMT
    '''
    x = tf.cond(tf.equal(dropout_prob, tf.constant(1.0)), lambda: input_data,
                lambda: tf.nn.dropout(input_data, dropout_prob + 0.3))
    # Initial Convolution Layer (comes before the ResNet modules)
    conv1 = conv2d_relu(x, weights['wcpre'], biases['bcpre'], strides=convStrides[0], pre_pad=convPaddings[1],
                        use_batchnorm=use_batchnorm, batchnorm_MA_frac=batch_norm_MA_decay)
    # ResNet module 1
    conv1 = resnet_module(conv1,
                          weights,
                          biases,
                          0,
                          resnet_n,
                          resnet_n_layers_per_block,
                          [convFiltSizes[0]] * resnet_n_layers_per_block,
                          subsample_on_first_block=False,
                          use_batchnorm=use_batchnorm,
                          batchnorm_MA_frac=batch_norm_MA_decay)
    # conv1 = tf.cond(tf.equal(classifier_id, tf.constant(-1)), lambda:conv1, lambda:tf.stop_gradient(conv1))
    # ResNet module 2
    conv2 = resnet_module(conv1,
                          weights,
                          biases,
                          1,
                          resnet_n,
                          resnet_n_layers_per_block,
                          [convFiltSizes[1]] * resnet_n_layers_per_block,
                          subsample_on_first_block=True,
                          use_batchnorm=use_batchnorm,
                          batchnorm_MA_frac=batch_norm_MA_decay)
    # conv2 = tf.cond(tf.equal(classifier_id, tf.constant(-1)), lambda:conv2, lambda:tf.stop_gradient(conv2))
    # Convolution Layer 3
    conv3 = resnet_module(conv2,
                          weights,
                          biases,
                          2,
                          resnet_n,
                          resnet_n_layers_per_block,
                          [convFiltSizes[2]] * resnet_n_layers_per_block,
                          subsample_on_first_block=True,
                          use_batchnorm=use_batchnorm,
                          batchnorm_MA_frac=batch_norm_MA_decay)
    # conv3 = tf.cond(tf.equal(classifier_id, tf.constant(-1)), lambda:conv3, lambda:tf.stop_gradient(conv3))
    ####### Fully connected layers ######
    fc2 = avgpool2d(conv3, averagePoolingFactor[2], averagePoolingFactor[2])  # Global average pooling
    fc2 = tf.reshape(fc2, [-1, weights['fc2_wout'].get_shape().as_list()[0]])
    logits = tf.add(tf.matmul(fc2, weights['fc2_wout']), biases['fc2_bout'])
    return logits


def resnet_construct_weights(num_resnet_modules, resnet_n, resnet_n_layers_per_block, image_channels, n_classes, convFiltSizes, convFiltNumbers, fcHiddenLayerOutputSz):
    '''
    Constructs two dictionaries "weights" and "biases" that contain all the trainable
    variables of the resnet model
    :param num_resnet_modules: usually 3.
    :param resnet_n: the number of resnet blocks in each module
    :param resnet_n_layers_per_block: number of the convolutional layers in every resnet block (usually 2 or 3)
    :param image_channels: the number of the input 3-rd dimension
    :param n_classes: the dimension of the output logits vector
    :param convFiltSizes: list of conv filter sizes. each size corresponds to a size of all filters in a a resnet moduile
    :param convFiltNumbers: list of conv filter numbers. each size corresponds to a size of all filters in a a resnet moduile
    :param fcHiddenLayerOutputSz: the number of the inputs to the FC layer
    :return: two dictionaries "weights" and "biases"
    '''

    # Construct layers weight & bias
    weights = {}
    biases = {}
    weights['wcpre'] = CONV2dVar(convFiltSizes[0], convFiltSizes[0], image_channels, convFiltNumbers[0], name='wcpre')
    biases['bcpre'] = BiasVar(convFiltNumbers[0], name='bcpre')
    for module in range(num_resnet_modules):

        # Construct resnet-blocks for clf_(module)
        for block in range(resnet_n):

            # Construct convolutional filters and biases in clf_(module)'s single block
            for layer in range(resnet_n_layers_per_block):
                weightName = "wc{}.{}.{}".format(module, block, layer)
                biasName = "bc{}.{}.{}".format(module, block, layer)

                if module > 0 and block == 0 and layer == 0:
                    weights[weightName] = CONV2dVar(convFiltSizes[module],
                                                    convFiltSizes[module],
                                                    convFiltNumbers[module - 1],
                                                    convFiltNumbers[module],
                                                    name=weightName)
                else:
                    weights[weightName] = CONV2dVar(convFiltSizes[module],
                                                    convFiltSizes[module],
                                                    convFiltNumbers[module],
                                                    convFiltNumbers[module],
                                                    name=weightName)
                biases[biasName] = BiasVar(convFiltNumbers[module], name=biasName)

        if module == num_resnet_modules - 1:
            fcWeightName = "fc{}_wout".format(module)
            fcBiasName = "fc{}_bout".format(module)
            weights[fcWeightName] = FCVar(fcHiddenLayerOutputSz, n_classes, name=fcWeightName)
            biases[fcBiasName] = BiasVar(n_classes, name=fcBiasName)

    return weights, biases


def fc_model(inputs, input_dim, output_dim, n_layers_fc, activation=tf.nn.tanh, dtype=tf.float32, slim=False):
    """
    Takes the "inputs" tensor and applies it to the "n_layers_fc"
    fully connected layers. This function constructs a computational graph for tensorflow.
    Use this function as a building block when defining the computational graph.

    default activation function: hyperbolic tangent.

    :param inputs:  a tensor of inputs to the model.
    :param input_dim: integer, expected input dimentions
    :param output_dim: integer, required number of outputs
    :param n_layers_fc: integer, specifying the number of the FC layers.

    :return: the output tensor of the model.
    """
    #fcHiddenLayerInputSz = [input_dim / (2**p) for p in range(n_layers_fc)]
    fcHiddenLayerInputSz = [input_dim for p in range(n_layers_fc)]

    if not slim:
        fcHiddenLayerInputSz = [input_dim for p in range(n_layers_fc)]
        fcHiddenLayerOutputSz = fcHiddenLayerInputSz[1:] + [output_dim]
    else:
        fcHiddenLayerInputSz = [input_dim] + [output_dim for p in range(n_layers_fc-1)]
        fcHiddenLayerOutputSz = fcHiddenLayerInputSz[1:] + [output_dim]
    outputs = inputs
    for layer_id in range(n_layers_fc):
        outputs = tf.layers.dense(outputs, fcHiddenLayerOutputSz[layer_id],
                                  activation=activation,
                                  kernel_initializer=tf.glorot_uniform_initializer(dtype=dtype),
                                  bias_initializer=tf.initializers.constant(dtype=dtype))
        if layer_id < n_layers_fc - 1:
            outputs = tf.layers.batch_normalization(outputs,
                                                    beta_initializer=tf.initializers.constant(dtype=dtype),
                                                    gamma_initializer=tf.initializers.constant(value=1,dtype=dtype),
                                                    moving_mean_initializer=tf.initializers.constant(dtype=dtype),
                                                    moving_variance_initializer=tf.initializers.constant(value=1,dtype=dtype))

    return outputs


# (num_all_measurements, lstm_output_dim, n_layers_lstm, input_data_X1)
def lstm_model(input, input_dim, output_dim, n_layers_lstm, dtype=tf.float32, dropout_keep_prob=1.0):
    """
       Create the LSTM modules stacked one after another.
       Return a list of lstm outputs (one element per
       lstm module) and a list of their final states.
    """
    lstmHiddenLayerInputSz = [input_dim / (2**p) for p in range(n_layers_lstm)]
    lstmHiddenLayerOutputSz = lstmHiddenLayerInputSz[1:] + [output_dim]
    #input = tf.unstack(input, None, 1)
    # import pdb
    # pdb.set_trace()
    lstms = [tf.nn.rnn_cell.LSTMCell(size, dtype=dtype, name='basic_lstm_cell_{}'.format(iii)) for iii,size in enumerate(lstmHiddenLayerOutputSz)]
    # Add dropout to the cell
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_keep_prob) for lstm in lstms]
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell(drops)
    # Getting an initial state of all zeros
    batch_size = tf.shape(input)[0]
    initial_state = cell.zero_state(batch_size, dtype)
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, input, initial_state=initial_state)
    return lstm_outputs, final_state
