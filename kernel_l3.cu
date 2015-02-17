__device__ void
gpu_init_temp (PROB_DATATYPE prob[NBETA_PER_WORD][NPROB_MAX], const int bidx)
{
  if (bidx < NPROB_MAX) {
    for (int b = 0; b < NBETA_PER_WORD; b++) {
      //prob[b][bidx] = 2.0f;
      prob[b][bidx] = UINT32_MAX;
    }
  }
}



// pre-compute propabilities
// load beta from global memory, compute, save propabilities in shared memory

/*
  bidx        0     1     2     3     4     5     6     7
  energy     -6+H  -6-H  -4+H  -4-H  -2+H  -2-H   0+H   0-H


  even
  energy = bidx - 6 + H
  ---------------------------------------
  bidx        0     2     4     6
  energy     -6+H  -4+H  -2+H   0+H


  odd
  energy = bidx - 7 - H = bidx - 6 + H - (1 + 2H)
  ----------------------------------------------
  bidx        1     3     5     7
  energy     -6-H  -4-H  -2-H   0-H
*/

__device__ void
  gpu_compute_temp
  (PROB_DATATYPE prob[NBETA_PER_WORD][NPROB_MAX], float *temp_beta_shared,
   const int bidx, int word)
{
  // for -2 < H < 2, it is OK to compute only first 8 elements of prob[14]
  // keep the rest unchanged


  if (bidx < NPROB) {
    for (int b = 0; b < NBETA_PER_WORD; b++) {
      float mybeta = temp_beta_shared[NBETA_PER_WORD * word + b];
      float energy = bidx - 6 - H - ((-1 * H * 2.0f + 1.0f) * (bidx & 1));

      //prob[b][bidx] = expf (2 * energy * mybeta);
      prob[b][bidx] = expf (2 * energy * mybeta) * UINT32_MAX;
    }
  }

}





// propose a temperature shuffle (inside a lattice)
__device__ void
gpu_shuffle (int *temp_idx, float *temp_beta, float *E, curandState * gpuseed,
	     const int bidx, int mod)
{

  int idx0, idx1, bidx_max;
  float delta_E, delta_beta;
  float myrand, val;
  int tmp0;
  float tmp1;


  curandState seed0 = gpuseed[BD * blockIdx.x + bidx];	// needed by curand

  __shared__ int __align__ (32) order[NBETA];

  if (bidx < NBETA)
    order[temp_idx[bidx]] = bidx;
  __syncthreads ();



  /*
     #ifdef DEBUG_PRINT_E
     if (blockIdx.x == 0 && bidx == 0) {
     for (int b = 0; b < NBETA; b++) {
     printf ("9 b=%02d %f \tE = %d\n", b, temp_beta[b], E[b]);
     }
     }
     #endif
   */



  if (mod == 0) {
    // 0 swap 1 , 2 swap 3 , 4 swap 5 , ...
    bidx_max = NBETA / 2;	// boundary
    idx0 = order[bidx << 1];
    idx1 = order[(bidx << 1) + 1];
  }
  else if (mod == 1) {
    // 0 , 1 swap 2 , 3 swap 4 , ...
    bidx_max = NBETA / 2 - 1;
    idx0 = order[(bidx << 1) + 1];
    idx1 = order[(bidx << 1) + 2];
  }

  if (bidx < bidx_max) {
    myrand = curand_uniform (&seed0);	// range: [0,1]
    delta_E = E[idx0] - E[idx1];
    delta_beta = temp_beta[idx0] - temp_beta[idx1];
    // test "expf" and "__expf" with extremely large input
    // verified their compatibities with positive infinity representation
    val = expf (delta_E * delta_beta);


    /*
       printf ("swap probability: %f =  exp (%f * %f) \t beta[%02d] = %f, beta[%02d] = %f \n",
       val, delta_E, delta_beta,
       idx0, temp_beta[idx0], idx1, temp_beta[idx1]);
     */

    // swap
    // branch would not hurt performance
    if (myrand < val) {
      tmp0 = temp_idx[idx0];
      temp_idx[idx0] = temp_idx[idx1];
      temp_idx[idx1] = tmp0;

      tmp1 = temp_beta[idx0];
      temp_beta[idx0] = temp_beta[idx1];
      temp_beta[idx1] = tmp1;
    }
  }

  gpuseed[BD * blockIdx.x + bidx] = seed0;
  __syncthreads ();
}






__device__ void
//__forceinline__
gpu_reduction (float *a, short a_shared[NBETA_PER_WORD][BD], const int bidx,
	       int word)
{
  // skewed sequential reduction is faster than tree reduction

#if 1
  // multi-threaded sequential reduction

  if (bidx < NBETA_PER_WORD) {
    int aaa = 0;
    for (int t = 0; t < BD; t++) {
      // skew loop iteration from "t" to "(t + bidx) % BD"
      // to avoid shared memory bank confict
      aaa += a_shared[bidx][(t + bidx) % BD];
    }

    // save the summation
    a[NBETA_PER_WORD * word + bidx] = aaa;
  }
#endif



#if 0
  // tree reduction

  __syncthreads ();

  //int powerof2 = power2floor (BD);

  for (int b = 0; b < NBETA_PER_WORD; b++) {
    /*
       for (int stride = BD / 2; stride >= 1; stride >>= 1) {
       if (bidx < stride)
       a_shared[b][bidx] += a_shared[b][stride + bidx];
       __syncthreads ();
       }
     */

    if (bidx < BD - powerof2) {
      a_shared[b][bidx] += a_shared[b][powerof2 + bidx];
    }
    for (int stride = powerof2 / 2; stride >= 1; stride >>= 1) {
      if (bidx < stride)
	a_shared[b][bidx] += a_shared[b][stride + bidx];
      __syncthreads ();
    }

  }

  // save the summation
  if (bidx < NBETA_PER_WORD)
    a[NBETA_PER_WORD * word + bidx] = (float) a_shared[bidx][0];
#endif

}
