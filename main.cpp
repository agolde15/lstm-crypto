#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <cudnn.h>
#include "cublas_v2.h"
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
using namespace std;

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

__global__ void MinMaxNormalize(float* input, float* output, const int newRange, const int oldRange, const int newMin, const int oldMin){
	int idx = threadIdx.x;
	output[idx] = ((((input[idx] - oldMin) * newRange) / oldRange) + newMin);
}


static size_t
getDeviceMemory(void) {
    struct cudaDeviceProp properties;
    int device;
    cudaError_t error;

    error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        fprintf(stderr, "failed to get device cudaError=%d\n", error);
        return 0;
    }

    error = cudaGetDeviceProperties(&properties, device);
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        fprintf(stderr, "failed to get properties cudaError=%d\n", error);
        return 0;
    }
    return properties.totalGlobalMem;
}

void readCsvFile(vector<vector<string>>& rows, std::string fileName, int asset){
	ifstream filein;
	filein.open(fileName);
	//error if file does not exist
	if (!boost::filesystem::exists(fileName)){
		cerr << "Input file " << fileName << " does not exist" << endl;
		exit(-1);
	}
	//error if not a csv input
	if (!boost::filesystem::extension(fileName).compare(".csv") == 0) {
		cerr << "Input file " << fileName << " must be CSV for training" << endl; 
		exit(-1);
	}
	//error if failed to open file
	if (filein.fail()){
		cerr << "Error opening input file: " << fileName << endl;
		exit(-1);
	}

	vector<string> row;
	string tempRow, value;
	
	// read header
	getline(filein,tempRow);
	
	// read lines from csv file
	while (!filein.eof()){
		getline(filein,tempRow);
		
		// separate row into delimited values
		stringstream str(tempRow);
		row.clear();
		while(getline(str, value, ',')){
			row.push_back(value);
		}
		if (stoi(row[1]) == asset){
			
			rows.push_back(row);
		}
	}
	
}

void createSeqLabels(vector<vector<string>>& input, vector<float*>& sequences, vector<float*>& labels, int seqSize, int labelSize, int feature, int splitIdx){
	float* temp;
	float* tempLabel;
	int temp_idx = 0;
	int temp_label= 0;
	cout << "Creating sequences and labels from " << splitIdx << " data points" << endl;

	//only segment on data before the train/test split
	for (int index = 0; index < (splitIdx - seqSize - labelSize); index+=labelSize){
		temp = new float[seqSize];
		temp_idx = 0;
		tempLabel = new float[labelSize];
		temp_label = 0; 
		
		for (int sub = index; sub < index + seqSize; sub++){
			temp[temp_idx] = stof(input[sub][feature]);
			temp_idx++; 
		}
		if (temp_idx == seqSize){
			sequences.push_back(temp);
		}	
	
		for (int sub = index + seqSize; sub < (index + seqSize + labelSize); sub++){
			tempLabel[temp_label] = stof(input[sub][feature]);
			temp_label++; 
		}
		if (temp_label == labelSize){
			labels.push_back(tempLabel);
		}	
	}
	
	cout << "Created " << sequences.size() << " sequences, " << labels.size() << " labels" << endl;
}


int main(int argc, char const *argv[]) {
	int batch = 1;
   	int inputSize = 1;
	int hiddenSize = 100;
	int numLayers = 1;
	int seqSize = 1440;
	int labelSize = 60;
	int features = 1;
	float splitPercent = 0.8;
	int feature = 3; // index in input data of the feature
 	if (argc < 3){
		cerr << "Usage: " << argv[0] << " <mode> <input>" << endl;
		exit(-1);
	}
	string procMode = argv[1];
	boost::to_upper(procMode);
 
	string inputName = argv[2];	
	int asset = 1;
	
	cublasStatus_t cbstatus;
	cublasHandle_t cbhandle;
	cublasCreate(&cbhandle);


	if (procMode.compare("TRAIN") == 0) {
		vector<vector<string>> rows;
		vector<vector<string>> subset;

		//read input data from CSV
		readCsvFile(rows, argv[2], asset);
		cout << "Read " << rows.size() << " input rows" << endl;
		for (int index = 0; index < 10; index++){
			std::cout << rows.front()[1] << std::endl;	

		}
	
		cout << "Created subset of data with asset " << asset << " containing " << rows.size() << " data points" << endl;	
		//remove original dataset

		//divide into train/test
		int splitIdx = (int)(splitPercent * rows.size());
		cout << "The train/test split index is " << splitIdx << endl;

		//extract floating point data for the feature
		cout << "Performing Min-Max normaliztation on training data" << endl;
		float* inputFeature = (float*)malloc(splitIdx*sizeof(float));
		float* outputFeature = (float*)malloc(splitIdx*sizeof(float));
		for (int index = 0; index < splitIdx; index++){
			inputFeature[index] = stof(rows[index][feature]);
		}
		
		float* cbInputFeature, devOutputFeature;
		cudaMalloc((void**)&cbInputFeature, splitIdx * sizeof(float));
		cudaMalloc((void**)&devOutputFeature, splitIdx * sizeof(float));
			
		int maxIndex, minIndex;
		int newMax = 1;
		int newMin = -1;
		float maxVal, minVal;
		cout << "Copying input feature data to device for CUBLAS" << endl;
		cudaMemcpy(cbInputFeature, inputFeature, splitIdx * sizeof(float), cudaMemcpyHostToDevice);
		
		//find max val
		cbstatus = cublasIsamax(cbhandle, splitIdx, cbInputFeature, 1, &maxIndex);
		if( cbstatus != CUBLAS_STATUS_SUCCESS){
			cerr << "CUBLAS error performing max" << endl;
			return(-1);
		}
		
		cudaMemcpy(&maxVal, cbInputFeature+maxIndex-1, sizeof(float), cudaMemcpyDeviceToHost);
		maxVal = (maxVal >= 0) ? maxVal : -maxVal;
		cout << "Absolute max of input is " << maxVal << endl;
		
		//find min val
		cbstatus = cublasIsamin(cbhandle, splitIdx, cbInputFeature, 1, &minIndex);
		if( cbstatus != CUBLAS_STATUS_SUCCESS){
			cerr << "CUBLAS error performing max" << endl;
			return(-1);
		}
		
		cudaMemcpy(&minVal, cbInputFeature+minIndex-1, sizeof(float), cudaMemcpyDeviceToHost);
		minVal = (minVal >= 0) ? minVal : -minVal;
		cout << "Absolute min of input is " << minVal << endl;

		/*
		Min-max normalization takes the absolute max/min of the original data and scales to a range
		of the new max/min.
		for a value x[n]:
		x[n]' = (x[n] - min) * (new_max - new_min)
				--------------------------------- + new_min
				 max - min 

		*/
		int blockSize = 256;
		int numBlocks = (splitIdx + blockSize -1) / blockSize
		MinMaxNormalize<<<numBlocks, blockSize>>>(cbInputFeature, devOutputFeature, (newMax - newMin), (maxVal - minVal), newMin, minVal);
		
		cudaMemcpy(&outputFeature, devOutputFeature, splitIdx * sizeof(float), cudaMemcpyDeviceToHost);
		
		free(inputFeature);
		cudaFree(cbInputFeature);
		cudaFree(devOutputFeature);
		cbstatus = cublasDestroy(cbhandle);
		
		if( cbstatus != CUBLAS_STATUS_SUCCESS){
			cerr << "CUBLAS shutdown error" << endl;
			return(-1);
		}
		
		//create training sequences and labels.
		vector<float*> sequences;
		vector<float*> labels;
		createSeqLabels(rows, sequences, labels, seqSize, labelSize, feature, splitIdx);
		
		


 	} else if (procMode.compare("INFER") == 0){
		cout << "Inference coming soon" << endl;
		
	} 
	/*
	cudnnHandle_t cudnnHandle;
	cudnnStatus_t status;
 	checkCUDNN(cudnnCreate(&cudnnHandle));
	cudnnRNNDataDescriptor_t RNNDataDesc;
    checkCUDNN(cudnnCreateRNNDataDescriptor(&RNNDataDesc));
	cudnnRNNMode_t = 
	cudnnRNNInputMode_t
	cudnnRNNBiatMode_t
	
	//checkCUDNN(cudnnRNNForward(&cudnnHandle, &RNNDataDesc, CUDNN_FWD_MODE_TRAINING));
    auto deviceMemoryAvailable  = getDeviceMemory();
	cout << deviceMemoryAvailable << endl;
	
	int inputTensorSize = seqSize * batch * inputSize;
	int outputTensorSize = seqSize * batch * inputSize;
	int hiddenTensorSize = numLayers * batch * hiddenSize;
	int hiddemDim[3]
	int hiddenDim[0] = numLayers;
	int hiddenDim[1] = batch;
	int hiddenDime[2] = hiddenSize;
 	int numLinearLayers = 8;
	int memoryUsage = (2 * inputTensorSize + 2 * outputTensorSize + 8 * hiddenTensorSize) * sizeof(float32);

	cudnnTensorDescriptor_t weightsDescriptor;
	cudnnTensorDescriptor_t biasDescriptor;
	cudnnTensorDescriptor_t inputDescriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&weightsDescriptor);
	checkCUDNN(cudnnCreateTensorDescriptor(&biasDescriptor);
	checkCUDNN(cudnnCreateTensorDescriptor(&inputDescriptor);

	const int nDims = 3; //batch, features, data
	int inputDims[nDims] = {batch, features, inputSize};
	checkCUDNN(cudnnSetTensorNdDescriptorEx(inputDescriptor, CUDA_DATA_FLOAT, inputDims));
	//allocate input array
	inputArr = (int*)malloc(batch * sizeof(int));
	cudaMalloc
	cudaMemcpy(inputArr, devInputArray, 

	checkCUDNN(cudnnDestroyTensorDescriptor(biasDescriptor);
	checkCUDNN(cudnnDestroyTensorDescriptor(weightsDescriptor);
	checkCUDNN(cudnnDestroyTensorDescriptor(inputDescriptor);
    cudnnDestroy(cudnnHandle);
	*/
}
