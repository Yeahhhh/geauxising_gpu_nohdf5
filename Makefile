
INC = -I. -I./include  -I./include/nvidia_gpucomputingsdk_4.2.9_c_common_inc
LIBDIR = -L/usr/local/packages/cuda/5.5.22/lib64
#LIBDIR = -L/opt/cuda/cuda-6.5/lib64

CC = gcc
CPP = g++
CXX = nvcc

#ARCHFLAG= -gencode arch=compute_20,code=sm_20
ARCHFLAG= -gencode arch=compute_35,code=sm_35 -Xptxas -dlcm=ca
#ARCHFLAG= -gencode arch=compute_50,code=sm_50 -Xptxas -dlcm=ca
OPTFLAG = -O3


CFLAGS = $(INC) -std=c99
CPPFLAGS = $(INC)
CXXFLAGS = $(INC) $(LIBDIR) $(ARCHFLAG)

srcdir = src
gpusrc = kernel.cu
cpusrc = host_main.cpp host_func.cpp host_launcher.cu

default: ising

%.o: %.c
	$(CC) $(OPTFLAG) $(CFLAGS) -c $<

%.o: %.cpp
	$(CPP) $(OPTFLAG) $(CPPFLAGS) -c $<

%.o: %.cu
	$(CXX) $(OPTFLAG) $(CXXFLAGS) -c $<

ising: host_main.o host_func.o host_launcher.o kernel.o
	$(CXX) $(OPTFLAG) $(CXXFLAGS) -o $@ $^

clean:
	rm -r *.o $(exec)

cleanoutput:
	rm -r output_*



