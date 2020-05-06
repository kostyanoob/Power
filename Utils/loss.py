import tensorflow as tf
from Utils.complex_numbers import complex_matmul_using_real_ops

def power_flow_equations_loss_real(sreal, simag, vreal, vimag, Yreal,Yimag, batch_size, dtype=tf.float32):
    """
    This function is using strictly real-values operators (not complex). The reason is
    that complex opearators were found not reliable in the presence of the tensorflow's
    backpropagation.
    finds the loss incurred by deviation from feasible power flow equations
    solution caused by the estimated voltages vector (v_est)

    :param sreal: active power vectors (batch_size x n_nodes x 1)
    :param simag: reactive power vectors (batch_size x n_nodes x 1)
    :param vreal: real part of the voltage vectors (batch_size x n_nodes x 1)
    :param vimag: imaginary part of the voltage vectors (batch_size x n_nodes x 1)
    :param Yreal: resistance matrix (n_nodes x n_nodes)
    :param Yimag: susceptance matrix (n_nodes x n_nodes)
    :param dtype: tensorflow type of number representation (tf.float32, tf.float64)
    :return: 2 Tensorflow OPs which compute:
            (s - diag(v_est) * conj(Y) * conj(v_est))^2  <-- averaged across all the examples in the batch
            Absolute(s - diag(v_est) * conj(Y) * conj(v_est))  <-- averaged across all the examples in the batch

    """
    # mse_cumulative = tf.zeros_like(vreal[0], dtype=dtype)
    # mae_cumulative = tf.zeros_like(vreal[0], dtype=dtype)
    # for i in range(batch_size):
    #
    #     diagVconjY_real, diagVconjY_imag = complex_matmul_using_real_ops(tf.diag(vreal[i]), tf.diag(vimag[i]), Yreal, Yimag, adjoint_b=True)
    #     diagVconjYconjV_real, diagVconjYconjV_imag = tf.squeeze(complex_matmul_using_real_ops(diagVconjY_real,
    #                                                                                           diagVconjY_imag,
    #                                                                                           tf.expand_dims(vreal[i], 0),
    #                                                                                           tf.expand_dims(vimag[i], 0), adjoint_b=True))
    #     diff_one_example_real, diff_one_example_imag = sreal[i] - diagVconjYconjV_real, simag[i] - diagVconjYconjV_imag
    #     mse_one_example = tf.square(diff_one_example_real) + tf.square(diff_one_example_imag)
    #     mae_one_example = tf.abs(diff_one_example_real) + tf.abs(diff_one_example_imag)
    #
    #     mse_cumulative += mse_one_example
    #     mae_cumulative += mae_one_example
    def complex_matmul_using_real_ops_diaga_adjY(packed_2_matrices):
        Areal, Aimag = packed_2_matrices
        Areal = tf.diag(Areal)
        Aimag = tf.diag(Aimag)
        return complex_matmul_using_real_ops(Areal, Aimag, Yreal, Yimag, adjoint_a=False, adjoint_b=True)

    def complex_matmul_using_real_ops_adjb_squeeze(packed_4_matrices):
        Areal, Aimag, Breal, Bimag = packed_4_matrices
        Preal, Pimag = complex_matmul_using_real_ops(Areal, Aimag, Breal, Bimag, adjoint_a=False, adjoint_b=True)
        return tf.squeeze(Preal), tf.squeeze(Pimag)

    diagVconjY_real, diagVconjY_imag = tf.map_fn(complex_matmul_using_real_ops_diaga_adjY, (vreal, vimag), dtype=(dtype,dtype), parallel_iterations=batch_size)
    vreal, vimag = tf.expand_dims(vreal, 1), tf.expand_dims(vimag, 1)
    diagVconjYconjV_real, diagVconjYconjV_imag = tf.map_fn(complex_matmul_using_real_ops_adjb_squeeze, (diagVconjY_real, diagVconjY_imag, vreal, vimag), dtype=(dtype,dtype), parallel_iterations=batch_size)
    diff_tensor_real, diff_tensor_imag = sreal - diagVconjYconjV_real, simag - diagVconjYconjV_imag
    msett = tf.reduce_mean(tf.square(diff_tensor_real) + tf.square(diff_tensor_imag))
    maett = tf.reduce_mean(tf.abs(diff_tensor_real) + tf.abs(diff_tensor_imag))
    return msett, maett


def power_flow_equations_loss_complex(s, v, Y, batch_size, dtype=tf.float32):
    """
    This function is using complex arithmetic, which was found unreliabel in the
    presence of tensorflow's backpropagation.
    finds the loss incurred by deviation from feasible power flow equations
    solution caused by the estimated voltages vector (v_est)

    :param s: complex vectors (batch_size x n_nodes x 1)
    :param v: complex vectors (batch_size x n_nodes x 1)
    :param Y: complex admittance matrix (n_nodes x n_nodes)
    :param dtype: tensorflow type of number representation (tf.float32, tf.float64)
    :return: 2 Tensorflow OPs which compute:
            (s - diag(v_est) * conj(Y) * conj(v_est))^2  <-- averaged across all the examples in the batch
            Absolute(s - diag(v_est) * conj(Y) * conj(v_est))  <-- averaged across all the examples in the batch

    """
    mse_cumulative = tf.zeros_like(s[0], dtype=dtype)
    mae_cumulative = tf.zeros_like(s[0], dtype=dtype)
    # TODO - speedup the matmul.
    for i in range(batch_size):

        diagVconjY = tf.matmul(tf.diag(v[i]), Y, adjoint_b=True, name='diagv_est_x_conjY')
        diagVconjYconjV = tf.squeeze(tf.matmul(diagVconjY, tf.expand_dims(v[i], 0), adjoint_b=True, name='diagv_est_x_conjY_x_conjv_est'))
        diff_one_example = s[i] - diagVconjYconjV
        mse_one_example = tf.square(tf.real(diff_one_example)) + tf.square(tf.imag(diff_one_example))
        mae_one_example = tf.abs(tf.real(diff_one_example)) + tf.abs(tf.imag(diff_one_example))
        # if i==0:
        #     retp = tf.print([tf.real(mse_one_example), tf.imag(mse_one_example)])
        mse_cumulative += mse_one_example
        mae_cumulative += mae_one_example
    return tf.reduce_mean(mse_cumulative)/batch_size, tf.reduce_mean(mae_cumulative)/batch_size #, retp
