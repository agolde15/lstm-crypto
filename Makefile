all: main.cu
	nvcc -L/usr/local/cuda/include -lcudart -lcuda -lcudnn -lcublas -lboost_filesystem main.cu -o main

