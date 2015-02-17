// initilize seeds for curand
// CURAND_Library.pdf, pp21
__global__ void
kernel_init_seed (int seed, Parameter para)
{
  const int gidx = BD * blockIdx.x + threadIdx.x;
  curand_init (seed, gidx, 0, &para.gpuseed[gidx]);
  // skipahead(100000, &para.gpuseed[gidx]);
}





__global__ void
kernel_warmup (Parameter para)
{
  const int bidx = threadIdx.x;

  /// temperature
  // (4 * 32) * 2 = 256 B
  __shared__ float __align__ (32) temp_beta_shared[NBETA];
  if (bidx < NBETA)
    temp_beta_shared[bidx] = para.temp[NBETA_MAX * blockIdx.x + bidx].beta;

  for (int i = 0; i < ITER_WARMUP_KERN; i += ITER_WARMUP_KERNFUNC) {
    mc (temp_beta_shared, para, ITER_WARMUP_KERNFUNC);
  }
}




__global__ void
kernel_swap (int rec, Parameter para)
{
  const int bidx = threadIdx.x;
  Temp *temp = para.temp;

  /// temperature
  // (4 * 32) * 2 = 256 B
  __shared__ int __align__ (32) temp_idx_shared[NBETA];
  __shared__ float __align__ (32) temp_beta_shared[NBETA];

  /// lattice energy
  // sizeof (float) * 32 = 128 B
  __shared__ float __align__ (32) E[NBETA];

  // load temperature
  if (bidx < NBETA) {
    temp_idx_shared[bidx] = temp[NBETA_MAX * blockIdx.x + bidx].idx;
    temp_beta_shared[bidx] = temp[NBETA_MAX * blockIdx.x + bidx].beta;
  }

  for (int i = 0; i < ITER_SWAP_KERN; i += ITER_SWAP_KERNFUNC) {
    int swap_mod = (i / ITER_SWAP_KERNFUNC) & 1;
    pt (temp_idx_shared, temp_beta_shared, E, para, swap_mod);
    mc (temp_beta_shared, para, ITER_SWAP_KERNFUNC);
  }

  // store temperature
  if (bidx < NBETA) {
    temp[NBETA_MAX * blockIdx.x + bidx].idx = temp_idx_shared[bidx];
    temp[NBETA_MAX * blockIdx.x + bidx].beta = temp_beta_shared[bidx];
  }
  __syncthreads ();

  // store energy status
  // if (bidx < NBETA)
  // para.st[rec].e[blockIdx.x][temp_idx_shared[bidx]] = E[bidx];
}










// rearrange the spins so that they matches the temperature order
// least significant bit - lowest temperature
// higher order bits - higher temperature
// for ACMSC

__global__ void
kernel_rearrange (Parameter para)
{
  const int bidx = threadIdx.x;
  MSC_DATATYPE *lattice = para.lattice;
  MSC_DATATYPE *lattice1 = para.lattice1;
  Temp *temp = para.temp;



  // temperature scratchpad
  __shared__ int __align__ (32) temp_idx_shared[NBETA_PER_WORD];

  // initilize lattice1
  for (int offset = 0; offset < SZ_CUBE; offset += BD)
    lattice1[SZ_CUBE * blockIdx.x + offset + bidx] = 0;
  __syncthreads ();


  int word = 0;
  int lattice_offset = (SZ_CUBE * NWORD * blockIdx.x) + (SZ_CUBE * word);

  if (bidx < NBETA_PER_WORD)
    temp_idx_shared[bidx] =
      temp[NBETA_MAX * blockIdx.x + NBETA_PER_WORD * word + bidx].idx;
  __syncthreads ();


  for (int offset = 0; offset < SZ_CUBE; offset += BD) {
    MSC_DATATYPE oldword = lattice[lattice_offset + offset + bidx];
    MSC_DATATYPE newword = 0;

    for (int i = 0; i < NSEG_PER_WORD; ++i) {
      for (int j = 0; j < NBETA_PER_SEG; ++j) {
	const int position = NBIT_PER_SEG * i + j;
	const int b = NBETA_PER_SEG * i + j;
	MSC_DATATYPE tmp = oldword >> position & 1;
	tmp <<= temp_idx_shared[b];
	newword |= tmp;
      }
    }
    lattice1[SZ_CUBE * blockIdx.x + offset + bidx] |= newword;
  }

}








__global__ void
kernel_compute_q (int rec, Parameter para)
{
  const int bidx = threadIdx.x;
  MSC_DATATYPE *lattice1 = para.lattice1;

  __shared__ MSC_DATATYPE l1[SZ_CUBE];
  __shared__ double qk_real[3][NBETA];
  __shared__ double qk_imag[3][NBETA];
  __shared__ double qk2_real[6][NBETA];
  __shared__ double qk2_imag[6][NBETA];
  const int lattice_offset0 = SZ_CUBE * (blockIdx.x << 1);
  const int lattice_offset1 = lattice_offset0 + SZ_CUBE;
  const double k = 2 * PI / L;

  for (int offset = 0; offset < SZ_CUBE; offset += BD) {
    l1[offset + bidx] =		// xord_word 
      lattice1[lattice_offset0 + offset + bidx] ^
      lattice1[lattice_offset1 + offset + bidx];
  }

  __syncthreads ();

  if (bidx < NBETA) {
    float q0 = 0.0f;
    for (int j = 0; j < 3; j++) {
      qk_real[j][bidx] = 0.0f;
      qk_imag[j][bidx] = 0.0f;
    }
    for (int j = 0; j < 6; j++) {
      qk2_real[j][bidx] = 0.0f;
      qk2_imag[j][bidx] = 0.0f;
    }

    MSC_DATATYPE xor_word;
    int xor_bit;

    for (int i = 0; i < SZ_CUBE; i++) {
      xor_word = l1[i];
      xor_bit = (xor_word >> bidx) & 0x1;
      xor_bit = 1 - (xor_bit << 1);	// parallel: +1, reverse: -1

      double bit = xor_bit;
      double x = i % L;
      double y = (i / L) % L;
      double z = (i / L) / L;
      /*      // 2 * pi / L * x_i
         angel1 = (double) (i % L) * 2 * PI / L;
         // 2 * pi / L * (x_i + y_i)
         angel2 = (double) (i % L + (i / L) % L) * 2 * PI / L;
       */
      q0 += bit;
      /*
         qk_real += (float)xor_bit * cos (angel1);
         qk_imag += (float)xor_bit * sin (angel1);
         qk2_real += (float)xor_bit * cos (angel2);
         qk2_imag += (float)xor_bit * sin (angel2);
       */
      qk_real[0][bidx] += bit * cos (x * k);
      qk_real[1][bidx] += bit * cos (y * k);
      qk_real[2][bidx] += bit * cos (z * k);

      qk_imag[0][bidx] += bit * sin (x * k);
      qk_imag[1][bidx] += bit * sin (y * k);
      qk_imag[2][bidx] += bit * sin (z * k);

      qk2_real[0][bidx] += bit * cos (x * k + y * k);
      qk2_real[1][bidx] += bit * cos (x * k - y * k);
      qk2_real[2][bidx] += bit * cos (x * k + z * k);
      qk2_real[3][bidx] += bit * cos (x * k - z * k);
      qk2_real[4][bidx] += bit * cos (y * k + z * k);
      qk2_real[5][bidx] += bit * cos (y * k - z * k);

      qk2_imag[0][bidx] += bit * sin (x * k + y * k);
      qk2_imag[1][bidx] += bit * sin (x * k - y * k);
      qk2_imag[2][bidx] += bit * sin (x * k + z * k);
      qk2_imag[3][bidx] += bit * sin (x * k - z * k);
      qk2_imag[4][bidx] += bit * sin (y * k + z * k);
      qk2_imag[5][bidx] += bit * sin (y * k - z * k);
    }


    St *st = para.st;

    // save measurements in "st"
    st[rec].q[blockIdx.x][bidx] = q0;
    for (int j = 0; j < 3; j++) {
      st[rec].qk_real[j][blockIdx.x][bidx] = qk_real[j][bidx];
      st[rec].qk_imag[j][blockIdx.x][bidx] = qk_imag[j][bidx];
    }
    //      (float) sqrt (qk_real * qk_real + qk_imag * qk_imag);
    for (int j = 0; j < 6; j++) {
      st[rec].qk2_real[j][blockIdx.x][bidx] = qk2_real[j][bidx];
      st[rec].qk2_imag[j][blockIdx.x][bidx] = qk2_imag[j][bidx];
    }
    //      (Float) sqrt (qk2_real * qk2_real + qk2_imag * qk2_imag);
  }
  __syncthreads ();
}
