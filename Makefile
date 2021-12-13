all: main.cpp
	nvcc -L/usr/local/cuda/include -lcudnn -lcublas -lboost_filesystem main.cpp -o main

