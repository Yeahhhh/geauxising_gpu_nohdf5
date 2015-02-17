#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "sim.h"
#include "sim.cuh"

#include "kernel_l3.cu"
#include "kernel_l2.cu"
#include "kernel_l1.cu"

