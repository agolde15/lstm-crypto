all: main.cpp
	nvcc -L/usr/local/cuda/include -lcudnn -lboost_filesystem main.cpp -o main

