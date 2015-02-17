#ifndef SIM_CUH
#define SIM_CUH




typedef struct
{
  curandState *gpuseed;
  MSC_DATATYPE *lattice;
  MSC_DATATYPE *lattice1;
  Temp *temp;
  St *st;
} Parameter;






// kernel_l3.cc
__device__ void gpu_init_temp (PROB_DATATYPE prob[NBETA_PER_WORD][NPROB_MAX], const int bidx);
__device__ void gpu_compute_temp (PROB_DATATYPE prob[NBETA_PER_WORD][NPROB_MAX], float *temp_beta_shared, const int bidx, int word);
__device__ void gpu_shuffle (int *temp_idx_shared, float *temp_beta_shared, float *E, curandState *gpuseed, const int bidx, int mod);
__device__ void gpu_reduction (float *a, short a_shared[NBETA_PER_WORD][BD], const int bidx, int word);


// kernel_l2.cu
__device__ void mc (float *temp_beta_shared, Parameter para, int iter);
__device__ void pt (int *temp_idx_shared, float *temp_beta_shared, float *E, Parameter para, int mod);


// kernel_l1.cu
__global__ void kernel_init_seed (int seed, Parameter para);
__global__ void kernel_warmup (Parameter para);
__global__ void kernel_swap (int rec, Parameter para);
__global__ void kernel_compute_q (int rec, Parameter para);
__global__ void kernel_rearrange (Parameter para);

#endif /* SIM_CUH */
