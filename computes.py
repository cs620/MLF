import cv2
import numba
import cmath
import math
import operator
from numba import vectorize, guvectorize, float32, cuda, jit
import cudaOperation as co
#==================================================================================
# Compute the innerproduct of the 'V'.
@guvectorize([(float32[:], float32[:])], '(n)->()', target = 'cuda')
def vectorProduce(V, res):
    for i in range(V.shape[0]):
        res[0] += V[i]*V[i]
#==================================================================================
# Compute the dot of the vector 'V1' and 'V2'.
@guvectorize([(float32[:], float32[:], float32[:])], '(n),(n)->()', target = 'cuda')
def dot(V1, V2, res):
    for i in range(V1.shape[0]):
        res[0] += V1[i]*V2[i]
# ==================================================================================
# Compute correlation between vector 'data' and 'label'.
@jit(float32(float32[:], float32[:]))
def Pierson(data, label):
    data = np.array(data, dtype='float32').reshape((-1, ))
    label = np.array(label, dtype='float32').reshape((-1, ))
    d = vectorProduce(data)[0]
    # print(d)
    d = cmath.sqrt(d)
    l = vectorProduce(label)[0]
    # print(l)
    l = cmath.sqrt(l)
    a = dot(data, label)[0]
    # print(d, l, a)
    return float(a/(d*l))
#==================================================================================
def conv(data, filter):
    """
    Return a result of the conv2d operate which produce from data and filter by CUDA.
    np.array :param data: input matrix.
    :param filter: implement conv2d operation over the 'data' array,
                    it should be odd number shape, and should be a square.
    :return: res.
            result of the conv2d.
    """
    h, w = data.shape
    res = np.zeros_like(data).astype(np.float32)
    gpu_res = cuda.to_device(res)
    gpu_data = cuda.to_device(data)
    gpu_filter = cuda.to_device(filter)
    center = math.floor(filter.shape[0]/2)
    co.cuConv2d[(math.ceil(h / 32),
                 math.ceil(w / 32)),
                (32, 32)](gpu_data, h, w, gpu_res, gpu_filter, center)
    res = gpu_res.copy_to_host()
    return res