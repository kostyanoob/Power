import scipy
import numpy as np
from Utils.complex_numbers import realImagFormatToComplex, complexFormatToRealImag

from Utils.data_structures import PartialObservabilityProblem
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug


def power_flow_residual_function_complex(Scomplex, Vcomplex, Ycomplex):
    """
    assume that N is the number of nodes

    :param kwargs: Scomplex - complex vector containing N complex powers
                   Vcomplex - complex vector representing the N complex voltages
                   Ycomplex - an NxN complex admittance matrix
    :return: the error (aka residual) of the powerflow solution given the voltages x and the powers Scomplex
    """
    rhs = np.matmul(np.matmul(np.diag(Vcomplex), np.conj(Ycomplex)), np.conj(Vcomplex))
    residuals = Scomplex - rhs
    return  residuals


def power_flow_residual_function_real(x, *args, **kwargs):
    """
    A wrapper function for the complex numbers-based residual
    computation for power flow equations. Assume that N is the number of
    nodes in the power grid.
    :param x: real,imag,real,imag formatted vector, containing (target-powers,target-voltages) (containing 2(N-Ns + N) items)
    :param args: none
    :param kwargs: Sobservable - real,imag,real,imag formatted vector of powers (containing (2Ns entries)
                   Yreal - an NxN real conductance matrix
                   Yimag - an NxN real susceptance matrix
                   W - weights vector for WLS, containing one of the following options;
                         (1) N items - in this case, both real part of the i^th squared residual and its squared imaginary part will be both multiplied by the same W[i]
                         (2) 2N items - in this case the real part of the i^th residual will be ultiplied by W[2i] and its imaginry part will be multiplied by W[2i+1]
                    power_reordering_indices_WLS - an array of indices that allows mapping [Sobservable,S-target] into a correctly ordered 1,...N bus order.
    :return: the error (aka residual) of the power-flow solution given the
             voltages x and the powers S
    """
    # Convert the powers, voltages and the admittances into their complex representation
    #ld("power_flow_residual_function_real : kwarg keys: {}".format(kwargs.keys()))
    N = kwargs['Yimag'].shape[-1]
    Ns = kwargs['Sobservable'].shape[-1]//2
    Ntarget_powers = N - Ns

    Srealimag_unordered = np.concatenate((kwargs['Sobservable'], x[:2*Ntarget_powers]))
    Srealimag = np.take(Srealimag_unordered, indices=kwargs['power_reordering_indices_WLS'])
    Scomplex = realImagFormatToComplex(Srealimag)
    Vcomplex = realImagFormatToComplex(x[2*Ntarget_powers:]) # The voltages are sitting right after the Ns powers
    Ycomplex = kwargs['Yreal'] + 1j * kwargs['Yimag']

    # Sanity checks
    assert 2 * len(Vcomplex.shape) == 2 * len(Scomplex.shape) == len(Ycomplex.shape) == 2
    assert Vcomplex.shape[0] == Scomplex.shape[0] == Ycomplex.shape[0] == Ycomplex.shape[1] == N

    # run the complex-based residual computation
    complex_residual = power_flow_residual_function_complex(Scomplex=Scomplex, Vcomplex=Vcomplex, Ycomplex=Ycomplex)

    # convert the complex residuals into real-valued residuals
    real_residual = complexFormatToRealImag(complex_residual)

    # WLS essence : Square the residual and multiply them by weights
    real_residual = np.square(real_residual)
    real_residual *= kwargs['W'].repeat(2) if len(kwargs['W'])==N else kwargs['W'] # The weights must be replicated twice to multiply the imaginary and the real part of each residual by the same weight
    return real_residual


def solve_wls_with_power(Sobservable : np.ndarray, S0: np.ndarray, V0 : np.ndarray,
                         Yreal: np.ndarray, Yimag: np.ndarray, weights_for_WLS : np.ndarray,
                         power_reordering_indices_WLS : np.ndarray):
    """
    :param Sobservable: real,imag,real,imag - formatted array.
                       Can be of the shape Bx2Ns if B examples
                       are to be computed in a single, batched,
                       call to this function.
    :param S0: real,imag,real,imag - formatted array.
                       Can be of the shape Bx2(N-Ns) if B examples
                       are to be computed in a single, batched,
                       call to this function.
    :param V0: real,imag,real,imag - formatted array.
               Can be of the shape Bx2N if B examples
               are to be computed in a single, batched,
               call to this function.
    :param Yreal: real valued conductance matrix matrix N x N
    :param Yimag: real valued susceptance matrix matrix N x N
    :param weights_for_WLS: real valued weights for the equations BxN
    :param power_reordering_indices_WLS: integer indices that could be used for gathering the
                                         [Sobservable, S0] array into properly ordered array.
                                         This is important because the bus indices within the
                                         Sobservable and S0 are not ordered from 1 to N.
    :return:
    """

    solver_argument_dict = dict(fun=power_flow_residual_function_real,
                                x0=np.concatenate((S0,V0),axis=-1),
                                verbose=0, # 0-silent, 1-report termination, 2-report progress.
                                method='trf',
                                ftol=3e-16,
                                xtol=3e-16,
                                gtol=3e-16,
                                max_nfev=10000, # maximum allowed residual function evaluations
                                kwargs={'Sobservable': Sobservable,
                                        'Yreal':Yreal,
                                        'Yimag':Yimag,
                                        'W':weights_for_WLS,
                                        'power_reordering_indices_WLS':power_reordering_indices_WLS})

    if len(Sobservable.shape)==2:
        Vsol = np.zeros(np.shape(V0), dtype=V0.dtype)
        for example_id in range(Sobservable.shape[0]):
            solver_argument_dict['x0'] = np.concatenate((S0[example_id],V0[example_id]),axis=-1)
            solver_argument_dict['kwargs']['Sobservable'] = Sobservable[example_id]
            solver_argument_dict['kwargs']['W'] = weights_for_WLS[example_id]
            Vsol[example_id] = scipy.optimize.least_squares(**solver_argument_dict).x[len(S0[example_id]):]
    else:
        Vsol = scipy.optimize.least_squares(**solver_argument_dict).x[len(S0):]

    return Vsol