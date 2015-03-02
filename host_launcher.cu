#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "include/yeah/cudautil.h"
#include "include/yeah/timing.h"

#include "sim.h"
#include "sim.cuh"


void
host_launcher (float beta_low, float beta_high, char *mydir, int node,
	       int device)
{
  // initilize random sequence
  srand (time (NULL) + 10 * node + device);

  // select a GPU device
  cudaSetDevice (device);

  // configure the GPU SRAM
  cudaFuncSetCacheConfig (kernel_warmup, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig (kernel_swap, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig (kernel_rearrange, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig (kernel_compute_q, cudaFuncCachePreferShared);


  // curand seeds
  curandState *gpuseed_dev;
  size_t gpuseed_sz = sizeof (curandState) * BD * GD;
  CUDAMALLOC (gpuseed_dev, gpuseed_sz, curandState *);

  // lattice
  MSC_DATATYPE *lattice, *lattice_dev;
  size_t lattice_sz = sizeof (MSC_DATATYPE) * SZ_CUBE * NWORD * GD;
  lattice = (MSC_DATATYPE *) malloc (lattice_sz);
  CUDAMALLOC (lattice_dev, lattice_sz, MSC_DATATYPE *);
  host_init_lattice (lattice);
  CUDAMEMCPY (lattice_dev, lattice, lattice_sz, cudaMemcpyHostToDevice);

  // lattice1
  // spins have been rearranged to reflect the temperature order
  MSC_DATATYPE *lattice1_dev;
  size_t lattice1_sz = sizeof (MSC_DATATYPE) * SZ_CUBE * GD;
  CUDAMALLOC (lattice1_dev, lattice1_sz, MSC_DATATYPE *);

  // temp - index and beta
  Temp *temp, *temp_dev;
  size_t temp_sz = sizeof (Temp) * NBETA_MAX * GD;
  temp = (Temp *) malloc (temp_sz);
  CUDAMALLOC (temp_dev, temp_sz, Temp *);
  host_init_temp (temp, beta_low, beta_high);
  CUDAMEMCPY (temp_dev, temp, temp_sz, cudaMemcpyHostToDevice);

  // st - status records
  St *st, *st_dev;
  size_t st_sz = sizeof (St) * ITER_SWAP / ITER_SWAP_KERN;
  st = (St *) malloc (st_sz);
  CUDAMALLOC (st_dev, st_sz, St *);
#ifdef DEBUG0
  printf ("st_sz = %f MB\n", (float) st_sz / 1024 / 1024);
#endif






  Parameter para;
  para.gpuseed = gpuseed_dev;
  para.lattice = lattice_dev;
  para.lattice1 = lattice1_dev; 
  para.temp = temp_dev;
  para.st = st_dev;









  // how often to re-initialize gpuseed???
  CUDAKERNELSYNC (kernel_init_seed, GD, BD, rand (), para);



  double t[4][2], t2[3], t3 = 0; // timing information

  char message[STR_LENG];

  putchar ('\n');
  host_report_speed_title ();


  // warm up runs
  t2[0] = HostTimeNow ();
  for (int i = 0; i < ITER_WARMUP; i += ITER_WARMUP_KERN) {
    t[0][0] = HostTimeNow ();

    CUDAKERNELSYNC (kernel_warmup, GD, BD, para);

    t[0][1] = HostTimeNow ();
    sprintf (message, "n%03d d%d warmup %8d/%08d", node, device, i, ITER_WARMUP);
    host_report_speed (t[0][0], t[0][1], ITER_WARMUP_KERN, message);
  }

  t2[1] = HostTimeNow ();

  // swap runs
  for (int i = 0; i < ITER_SWAP; i += ITER_SWAP_KERN) {
    t[1][0] = HostTimeNow ();

    CUDAKERNELSYNC (kernel_swap, GD, BD, i / ITER_SWAP_KERN, para);

    t[1][1] = HostTimeNow ();
    t3 += t[1][1] - t[1][0];

    CUDAKERNELSYNC (kernel_rearrange, GD, BD, para);
    CUDAKERNELSYNC (kernel_compute_q, GD_HF, BD, i / ITER_SWAP_KERN, para);

    t[2][1] = HostTimeNow ();

    sprintf (message, "n%03d d%d PT     %8d/%08d", node, device, i, ITER_SWAP);
    host_report_speed (t[1][0], t[1][1], ITER_SWAP_KERN, message);
  }
  t2[2] = HostTimeNow ();



#ifndef NO_OUTPUT
  CUDAMEMCPY (st, st_dev, st_sz, cudaMemcpyDeviceToHost);
  host_save_st (st, mydir, node, device);
#endif



  // report overall speed
  putchar ('\n');
  sprintf (message, "n%03d d%d overall warmup          ", node, device);
  host_report_speed (t2[0], t2[1], ITER_WARMUP, message);
  sprintf (message, "n%03d d%d overall PT (no measure) ", node, device);
  host_report_speed (0, t3, ITER_SWAP, message);
  sprintf (message, "n%03d d%d overall PT              ", node, device);
  host_report_speed (t2[1], t2[2], ITER_SWAP, message);
  sprintf (message, "n%03d d%d overall simulation      ", node, device);
  host_report_speed (t2[0], t2[2], ITER_WARMUP + ITER_SWAP, message);
  putchar ('\n');


  CUDAFREE (gpuseed_dev);
  CUDAFREE (lattice_dev);
  CUDAFREE (lattice1_dev);
  CUDAFREE (temp_dev);
  CUDAFREE (st_dev);

  free (lattice);
  free (temp);
  free (st);
}

