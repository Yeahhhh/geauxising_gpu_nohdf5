.SUFFIXES: .cu .cuh .h

INC = -I. -I./include  -I./include/nvidia_gpucomputingsdk_4.2.9_c_common_inc
LIBDIR = -L/opt/cuda/cuda-6.5/lib64
LIB = -lcudart

CC = gcc
CPP = g++
CXX = nvcc
MPICC = h5pcc

ARCHFLAG= -gencode=arch=compute_50,code=sm_50
OPTFLAG0 = -O0 -g -G
OPTFLAG1 = -O0
OPTFLAG2 = -O2
OPTFLAG3 = -O3
PROFFLAG = --ptxas-options=-v -keep

CFLAGS = $(INC) -std=c99
CPPFLAGS = $(INC)
CXXFLAGS = $(INC) $(LIB) $(LIBDIR) $(ARCHFLAG)

srcdir = src
gpusrc = kernel.cu
cpusrc = host_main.cpp host_func.cpp host_launcher.cu
exec = ising

default: $(exec) 


#.cu: Makefile
#	$(CXX) $(CFLAGS) -o $@ $*.cu
#


%.o: %.c
	$(CC) $(OPTFLAG2) $(CFLAGS) -c $<

%.o: %.cpp
	$(CPP) $(OPTFLAG2) $(CPPFLAGS) -c $<

%.o: %.cu
	$(CXX) $(OPTFLAG2) $(CXXFLAGS) -c $<

mpiprocess.o: mpiprocess.cpp
	$(MPICC) -c $<

mpi_ising: host_main.o mpiprocess.o host_func.o host_launcher.o kernel.o
	$(MPICC) $(OPTFLAG2) $(LIB) $(LIBDIR) -o $@ $^

ising: host_main.o host_func.o host_launcher.o kernel.o
	$(CXX) $(OPTFLAG2) $(CXXFLAGS) -o $@ $^


prof: $(gpusrc) $(cpusrc)
	$(CXX) $(OPTFLAG1) $(CXXFLAGS) $(PROFFLAG) $^

profclean: $(gpusrc) $(cpusrc)
	$(CXX) $(OPTFLAG1) $(CXXFLAGS) $(PROFFLAG) -clean $^

g_ising: $(gpusrc) $(cpusrc)
	$(CXX) $(OPTFLAG0) $(CXXFLAGS) -o $@ $^

clean:
	rm -r *.o $(exec)

cleanoutput:
	rm -r output_*



