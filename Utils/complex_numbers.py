import numpy as np
import tensorflow as tf
def realImagVectorsToMagAngVectors(realVec, imagVec, deg=True):
    """
    Receives two real valued vectors x and y, which 
    represents the complex vector x+1j*y.
    
    :param realVec: numpy array / list of real values
    :param imagVec: numpy array / list of imaginary values
    :param deg: Boolean, set to truth if degrees are 
                the units of the angle/ If set to False, 
                then radians will be used for the output 
                angle vector.
    :return: two real valued vectors that represents the Magnitude(x+1j*y) and Angle(x+1j*y)
             same type as the input arrays.
            
    """
    assert(type(realVec) == type(imagVec))
    if type(realVec)==list:
        retType = list
        imagVec = np.array(imagVec)
        realVec = np.array(realVec)
    else:
        retType = np.array
    assert(realVec.shape == imagVec.shape)

    complex_vec = realVec + 1j * imagVec
    magnitude_vec = np.absolute(complex_vec)
    angle_vec = np.angle(complex_vec, deg=deg)
    return retType(magnitude_vec), retType(angle_vec)

def complex_matmul_using_real_ops(Areal, Aimag, Breal, Bimag, adjoint_a=False, adjoint_b=False):
    """
     P=A*B matrix product OP. where instead of complex matrices we use
     real valued representation as follows:

        A=Areal + 1j * Aimag
        B=Breal + 1j * Bimag
        P=Preal + 1j * Pimag

    :param Areal: tensorflow tensor of real valued numbers
    :param Aimag:
    :param Breal:
    :param Bimag:
    :return: tuple(Preal, Pimag)
    """

    if adjoint_a:
        Areal = tf.transpose(Areal)
        Aimag = tf.transpose(tf.negative(Aimag))

    if adjoint_b:
        Breal = tf.transpose(Breal)
        Bimag = tf.transpose(tf.negative(Bimag))

    Preal = tf.subtract(tf.matmul(Areal,Breal), tf.matmul(Aimag,Bimag))
    Pimag = tf.add(tf.matmul(Areal, Bimag), tf.matmul(Aimag, Breal))

    return Preal, Pimag

def realImagFormatToComplex(realImagFormattedVec):
    """
    A reformatting function which converts a real valued array into complex vector
    Let the values in the input array be [r1, i1, r2, i2,..., rN, iN]
    then the output array will be [r1+j*i1,r 2+j*i2,..., rN+j*iN]
    :param realImagFormattedVec:
    :return: returns a 1D numpy array of a length N containing complex numbers.
    """
    twoN = len(realImagFormattedVec)
    real_vals = np.take(realImagFormattedVec, range(0,twoN,2))
    imag_vals = np.take(realImagFormattedVec, range(1, twoN, 2))
    return real_vals + 1j * imag_vals


def complexFormatToRealImag(complexVec):
    """
    A reformatting function which converts a complex vector into real valued array.
    Let the values in the input array be [r1+j*i1,r 2+j*i2,..., rN+j*iN]
    then the output array will be [r1, i1, r2, i2,..., rN, iN]
    :param complexVec: complex numpy ndarray
    :return: returns a 1D numpy array of a length N containing complex numbers.
    """
    N = len(complexVec)
    ret = np.empty((2*N,), dtype=np.real(complexVec).dtype)
    ret[0::2] = complexVec.real
    ret[1::2] = complexVec.imag
    return ret


