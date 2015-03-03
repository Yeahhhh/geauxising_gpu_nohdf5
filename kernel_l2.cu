__device__ void
mc (float *temp_beta_shared, Parameter para, int iter)
{
  const int bidx = threadIdx.x;
  MSC_DATATYPE *lattice = para.lattice;
  curandState seed0 = para.gpuseed[BD * blockIdx.x + bidx];


  /// temperature scratchpad
  __shared__ PROB_DATATYPE temp_prob_shared[NBETA_PER_WORD][NPROB_MAX];
  gpu_init_temp (temp_prob_shared, bidx);

  /// lattice scratchpad
  // sizeof(int32_t) * 16 * 16 * 16 = 16 KB
  __shared__ MSC_DATATYPE l[SZ_CUBE];



  // 3D thread dimensions
  const int bdx = L / 2;	// blockDim.x
  const int bdy = L;		// blockDim.y
  const int bdz = blockDim.x / bdx / bdy;	// blockDim.z

  // 3D thread index
  const int tz = threadIdx.x / bdx / bdy;
  const int ty = (threadIdx.x - bdx * bdy * tz) / bdx;
  const int tx = threadIdx.x - bdx * bdy * tz - bdx * ty;

  // map threads to lattice points
  const int y = ty;
  const int ya = (y + L - 1) % L;
  const int yb = (y + 1) % L;


  for (int word = 0; word < NWORD; word++) {
    int lattice_offset = SZ_CUBE * NWORD * blockIdx.x + SZ_CUBE * word;

    // initilize temperature scratchpad
    gpu_compute_temp (temp_prob_shared, temp_beta_shared, bidx, word);

    // import lattice scratchpad
    for (int idx = bidx; idx < SZ_CUBE; idx += BD) {
      l[idx] = lattice[lattice_offset + idx];
    }
    __syncthreads ();



    for (int i = 0; i < iter; i++) {

      // two phases update
      for (int run = 0; run < 2; run++) {
	const int x = (tx << 1) + ((tz & 1) ^ (ty & 1) ^ run);
	const int xa = (x + L - 1) % L;
	const int xb = (x + 1) % L;
	//const int xa = (x + L - 1) & (L - 1);
	//const int xb = (x + 1) & (L - 1);

// attention
#pragma unroll 4
	for (int z = tz; z < L; z += bdz) {
	  const int za = (z + L - 1) % L;
	  const int zb = (z + 1) % L;
	  //const int za = (z + L - 1) & (L - 1);
	  //const int zb = (z + 1) & (L - 1);

	  MSC_DATATYPE c = l[CUBEIDX (z, y, x)];	// center
	  MSC_DATATYPE n0 = l[CUBEIDX (z, y, xa)];	// left
	  MSC_DATATYPE n1 = l[CUBEIDX (z, y, xb)];	// right
	  MSC_DATATYPE n2 = l[CUBEIDX (z, ya, x)];	// up
	  MSC_DATATYPE n3 = l[CUBEIDX (z, yb, x)];	// down
	  MSC_DATATYPE n4 = l[CUBEIDX (za, y, x)];	// front
	  MSC_DATATYPE n5 = l[CUBEIDX (zb, y, x)];	// back

	  // for profiling purpose
	  //float val = 0.7;
	  //float myrand = curand_uniform (&seed0);
	  //PROB_DATATYPE myrand = 0.4;
	  //PROB_DATATYPE myrand = curand (&seed0);     // range: [0,UINT32_MAX]
	  //c = c ^ n0 ^ n1 ^ n2 ^ n3 ^ n4 ^ n5;


	  n0 = MASK_A * ((c >> SHIFT_J0) & 1) ^ n0 ^ c;
	  n1 = MASK_A * ((c >> SHIFT_J1) & 1) ^ n1 ^ c;
	  n2 = MASK_A * ((c >> SHIFT_J2) & 1) ^ n2 ^ c;
	  n3 = MASK_A * ((c >> SHIFT_J3) & 1) ^ n3 ^ c;
	  n4 = MASK_A * ((c >> SHIFT_J4) & 1) ^ n4 ^ c;
	  n5 = MASK_A * ((c >> SHIFT_J5) & 1) ^ n5 ^ c;

	  for (int s = 0; s < NBETA_PER_SEG; ++s) {
	    MSC_DATATYPE e =
	      ((n0 >> s) & MASK_S) +
	      ((n1 >> s) & MASK_S) +
	      ((n2 >> s) & MASK_S) +
	      ((n3 >> s) & MASK_S) +
	      ((n4 >> s) & MASK_S) + ((n5 >> s) & MASK_S);
	    e = (e << 1) + ((c >> s) & MASK_S);
	    MSC_DATATYPE flip = 0;

//	    #pragma unroll
	    for (int seg_offset = 0; seg_offset < SHIFT_MAX;
		 seg_offset += NBIT_PER_SEG) {
	      PROB_DATATYPE val =
		temp_prob_shared[seg_offset + s][(e >> seg_offset) & MASK_E];
	      PROB_DATATYPE myrand = curand (&seed0);	// range: [0,UINT32_MAX]
	      flip |= ((MSC_DATATYPE) (myrand < val) << seg_offset);	// myrand < val ? 1 : 0;
	    }
	    c ^= (flip << s);
	  }

	  l[CUBEIDX (z, y, x)] = c;
	}			// z

	__syncthreads ();
      }				// run
    }				// i

    // export lattice scratchpad
    for (int idx = bidx; idx < SZ_CUBE; idx += BD) {
      lattice[lattice_offset + idx] = l[idx];
    }

    __syncthreads ();
  }				// word

  // copy back seed
  para.gpuseed[BD * blockIdx.x + bidx] = seed0;
}






__device__ void
pt (int *temp_idx_shared, float *temp_beta_shared, float *E, Parameter para,
    int mod)
{
  const int bidx = threadIdx.x;

  MSC_DATATYPE *lattice = para.lattice;


  /// E scratchpads
  // does "short" datatype degrade performance?

  // signed 16 bit integer: -32K ~ 32K, never overflows
  // sizeof (shot) * 24 * 512 = 24 KB
  __shared__ short E_shared[NBETA_PER_WORD][BD];
  //short E_shared[NBETA_PER_WORD][BD];
  // sizeof (float) * 32 = 128 B
  __shared__ float __align__ (32) Eh[NBETA];


  /// lattice scratchpad
  // sizeof (int32_t) * 16 * 16 * 16 = 16 KB
  __shared__ MSC_DATATYPE l[SZ_CUBE];



  // 3D thread dimensions
  const int bdx = L;		// blockDim.x
  const int bdy = L;		// blockDim.y
  const int bdz = blockDim.x / bdx / bdy;	// blockDim.z

  // 3D thread index
  const int tz = threadIdx.x / bdx / bdy;
  const int ty = (threadIdx.x - bdx * bdy * tz) / bdx;
  const int tx = threadIdx.x - bdx * bdy * tz - bdx * ty;

  // map threads to lattice points
  const int y = ty;
  const int ya = (y + L - 1) % L;
  const int yb = (y + 1) % L;
  const int x = tx;
  const int xa = (x + L - 1) % L;
  const int xb = (x + 1) % L;



  for (int word = 0; word < NWORD; word++) {
    int lattice_offset = SZ_CUBE * NWORD * blockIdx.x + SZ_CUBE * word;

    // import lattice scratchpad
    for (int idx = bidx; idx < SZ_CUBE; idx += BD) {
      l[idx] = lattice[lattice_offset + idx];
    }

    // reset partial status
    for (int b = 0; b < NBETA_PER_WORD; b++)
      E_shared[b][bidx] = 0;

    __syncthreads ();


// attention
#pragma unroll 4
    for (int z = tz; z < L; z += bdz) {
      int za = (z + L - 1) % L;
      int zb = (z + 1) % L;
      //int za = (z + L - 1) & (L - 1);
      //int zb = (z + 1) & (L - 1);

      MSC_DATATYPE c = l[CUBEIDX (z, y, x)];	// center
      MSC_DATATYPE n0 = l[CUBEIDX (z, y, xa)];	// left
      MSC_DATATYPE n1 = l[CUBEIDX (z, y, xb)];	// right
      MSC_DATATYPE n2 = l[CUBEIDX (z, ya, x)];	// up
      MSC_DATATYPE n3 = l[CUBEIDX (z, yb, x)];	// down
      MSC_DATATYPE n4 = l[CUBEIDX (za, y, x)];	// front
      MSC_DATATYPE n5 = l[CUBEIDX (zb, y, x)];	// back

      n0 = MASK_A * ((c >> SHIFT_J0) & 1) ^ n0 ^ c;
      n1 = MASK_A * ((c >> SHIFT_J1) & 1) ^ n1 ^ c;
      n2 = MASK_A * ((c >> SHIFT_J2) & 1) ^ n2 ^ c;
      n3 = MASK_A * ((c >> SHIFT_J3) & 1) ^ n3 ^ c;
      n4 = MASK_A * ((c >> SHIFT_J4) & 1) ^ n4 ^ c;
      n5 = MASK_A * ((c >> SHIFT_J5) & 1) ^ n5 ^ c;

      for (int s = 0; s < NBETA_PER_SEG; s++) {
	MSC_DATATYPE e =
	  ((n0 >> s) & MASK_S) +
	  ((n1 >> s) & MASK_S) +
	  ((n2 >> s) & MASK_S) +
	  ((n3 >> s) & MASK_S) + ((n4 >> s) & MASK_S) + ((n5 >> s) & MASK_S);
	//#pragma unroll
	for (int seg_offset = 0; seg_offset < SHIFT_MAX;
	     seg_offset += NBIT_PER_SEG) {
	  E_shared[seg_offset + s][bidx] += (e >> seg_offset) & MASK_E;	// range: [0,6]
	}
      }

    }				// z

    gpu_reduction (E, E_shared, bidx, word);
    __syncthreads ();



    /// energy contribute by external field

    for (int b = 0; b < NBETA_PER_WORD; b++)
      E_shared[b][bidx] = 0;

    for (int z = tz; z < L; z += bdz) {
      MSC_DATATYPE c = l[CUBEIDX (z, y, x)];

      for (int i = 0; i < NSEG_PER_WORD; ++i) {
	for (int j = 0; j < NBETA_PER_SEG; ++j) {
	  const int position = NBIT_PER_SEG * i + j;
	  E_shared[position][bidx] += ((c >> position) & 1);
	}
      }
    }

    gpu_reduction (Eh, E_shared, bidx, word);
    __syncthreads ();

  }				// word;



  // convert E from [0,6] to [-6,6], e = e * 2 - 6
  // E = sum_BD sum_ZITER (e * 2 - 6)
  //   = 2 * sum_ZITER_TperG e - 6 * ZITER * BD

  // conver Eh from [0,1] to [-1,1], e = e * 2 - 1
  // Eh = 2 * sum_ZITER_TperG e - SZ_CUBE

  // should not substrasct the constant

  if (bidx < NBETA) {
    E[bidx] = E[bidx] * 2 - 6 * SZ_CUBE;
    Eh[bidx] = Eh[bidx] * 2 - SZ_CUBE;
    E[bidx] = E[bidx] + Eh[bidx] * H;
  }
  __syncthreads ();

  gpu_shuffle (temp_idx_shared, temp_beta_shared, E, para.gpuseed, bidx, mod);
}
