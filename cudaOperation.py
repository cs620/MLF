import numba.cuda as cuda
import math
import numpy

@cuda.jit
def cuConv2d(data, h, w, res, filter, filter_center):
    """
    Return a result of the conv2d operate which produce from data and filter by CUDA.
    args:
        np.array :param data: input 2d matrix.
        int32 :param h: h of the data.
        int32 :param w: w of the data.
        np.array :param res: return result of the conv2d, got similar shape with data.
        np.array :param filter: filter operate over the data.
        int32 :param filter_center: center index of the filter.
    :return: res.
    """
    id = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x + (cuda.blockIdx.y + (cuda.gridDim.y * cuda.blockIdx.x)) * (cuda.blockDim.x * cuda.blockDim.y)
    fidx_x = filter_center
    fidx_y = filter_center
    idx_x = int(math.floor(id / w))
    idx_y = id % w
    if idx_y > 0 and idx_x > 0 and idx_x < h - 1 and idx_y < w - 1:
        res[idx_x][idx_y] = (data[idx_x][idx_y] * filter[fidx_x][fidx_y] + data[idx_x - 1][idx_y - 1] * filter[fidx_x - 1][fidx_y - 1] + \
                            data[idx_x - 1][idx_y] * filter[fidx_x - 1][fidx_y] + data[idx_x][idx_y - 1] * filter[fidx_x][fidx_y - 1] + data[idx_x + 1][idx_y - 1] * \
                            filter[fidx_x + 1][fidx_y - 1] + \
                            data[idx_x - 1][idx_y + 1] * filter[fidx_x - 1][fidx_y + 1] + data[idx_x][idx_y + 1] * filter[fidx_x][fidx_y + 1] + \
                            data[idx_x + 1][idx_y + 1] * filter[fidx_x + 1][fidx_y + 1]+ \
                            data[idx_x + 1][idx_y] * filter[fidx_x + 1][fidx_y])