import scipy
import numpy as np
from Utils.complex_numbers import realImagFormatToComplex, complexFormatToRealImag

from Utils.data_structures import PartialObservabilityProblem
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug


def power_flow_residual_function_complex(x, *args, **kwargs):
    """
    assume that N is the number of nodes
    :param x: complex vector representing the complex voltages
    :param args: none
    :param kwargs: Scomplex - complex vector containing N complex powers
                   Ycomplex - an NxN complex admittance matrix
                   W - weights matrix for WLS Fix: W shouldn't be applied within this function!
    :return: the error (aka residual) of the powerflow solution given the voltages x and the powers Scomplex
    """
    #ld("power_flow_residual_function_complex : kwarg keys: {}".format(kwargs.keys()))
    Scomplex = kwargs['Scomplex']
    Ycomplex = kwargs['Ycomplex']
    rhs = np.matmul(np.matmul(np.diag(x), np.conj(Ycomplex)), np.conj(x))
    residuals = Scomplex - rhs
    return  residuals


def power_flow_residual_function_real(x, *args, **kwargs):
    """
    A wrapper function for the complex numbers-based residual
    computation for power flow equations. Assume that N is the number of
    nodes in the power grid.
    :param x: real,imag,real,imag formatted vector of voltages (containing 2N items)
    :param args: none
    :param kwargs: S - real,imag,real,imag formatted vector of voltages (containing (2N entries)
                   Yreal - an NxN real conductance matrix
                   Yimag - an NxN real susceptance matrix
                   W - weights vector for WLS, containing one of the following options;
                         (1) N items - in this case, both real part of the i^th squared residual and its squared imaginary part will be both multiplied by the same W[i]
                         (2) 2N items - in this case the real part of the i^th residual will be ultiplied by W[2i] and its imaginry part will be multiplied by W[2i+1]
    :return: the error (aka residual) of the power-flow solution given the
             voltages x and the powers S
    """
    # Convert the powers, voltages and the admittances into their complex representation
    #ld("power_flow_residual_function_real : kwarg keys: {}".format(kwargs.keys()))
    Scomplex = realImagFormatToComplex(kwargs['S'])
    Ycomplex = kwargs['Yreal'] + 1j * kwargs['Yimag']
    x = realImagFormatToComplex(x)

    # run the complex-based residual computation
    complex_residual = power_flow_residual_function_complex(x, Scomplex=Scomplex, Ycomplex=Ycomplex, W=kwargs['W'])

    # convert the complex residuals into real-valued residuals
    real_residual = complexFormatToRealImag(complex_residual)

    # WLS essence : Square the residual and multiply them by weights
    real_residual = np.square(real_residual) # TODO: according to the scipy.least_squares documentation, the squaring will be performed by the optimzier itself. Maybe this shouldn't be applies here!
    real_residual *= kwargs['W'].repeat(2) if len(kwargs['W'])==N else kwargs['W'] # The weights must be replicated twice to multiply the imaginary and the real part of each residual by the same weight
    return real_residual


def solve_wls(S : np.ndarray,V0 : np.ndarray, Yreal: np.ndarray, Yimag: np.ndarray, weights_for_WLS : np.ndarray):
    """

    :param S: real,imag,real,imag - formatted array.
               Can be of the shape BxN if B examples
               are to be computed in a single, batched,
               call to thi s function.
    :param V0: real,imag,real,imag - formatted array.
               Can be of the shape BxN if B examples
               are to be computed in a single, batched,
               call to thi s function.
    :param Yreal: real valued conductance matrix matrix N/2 x N/2
    :param Yimag: real valued susceptance matrix matrix N/2 x N/2
    :param weights_for_WLS: real valued weights for the equations Bx N/2
    :return:
    """

    solver_argument_dict = dict(fun=power_flow_residual_function_real,
                                x0=V0,
                                verbose=0, # 0-silent, 1-report termination, 2-report progress.
                                method='lm',
                                ftol=3e-16,
                                xtol=3e-16,
                                gtol=3e-16,
                                max_nfev=10000, # maximum allowed residual function evaluations
                                kwargs={'S': S,
                                        'Yreal':Yreal,
                                        'Yimag':Yimag,
                                        'W':weights_for_WLS})

    if len(S.shape)==2:
        Vsol = np.zeros(np.shape(V0), dtype=V0.dtype)
        for example_id in range(S.shape[0]):
            solver_argument_dict['x0'] = V0[example_id]
            solver_argument_dict['kwargs']['S'] = S[example_id]
            solver_argument_dict['kwargs']['W'] = weights_for_WLS[example_id]
            Vsol[example_id] = scipy.optimize.least_squares(**solver_argument_dict).x
    else:
        Vsol = scipy.optimize.least_squares(**solver_argument_dict).x

    return Vsol