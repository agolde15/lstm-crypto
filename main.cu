/*
main.cu

This file uses CUDA, cuBLAS and cuDNN to build an LSTM
model for the G-Research Crypto Forecasting dataset.

The cuDNN initialization is follows, roughly, the structure of the 
NVIDIA RNN sample code:
https://developer.nvidia.com/discover/lstm
*/
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


//This will check a cudnnStatus_t object for failures
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

/* Kernel to perform MinMax Normalization
	Min-max normalization takes the absolute max/min of the original data and scales to a range
	of the new max/min.
	for a value x[n]:
	x[n]' = (x[n] - min) * (new_max - new_min)
			--------------------------------- + new_min
			 max - min 
*/
__global__ void minmax(float* input, float* output, float newRange, float oldRange, float newMin, float oldMin){
	int idx = (blockIdx.x * blockDim.x) +  threadIdx.x;
	output[idx] = ((((input[idx] - oldMin) * newRange) / oldRange) + newMin);
}

// Method to read in training data from a csv file
// vector<vector<string>>& rows -> pointer to vector to store input data
// string filename -> file name to read (CSV)
// asset -> integer representing asset ID to read
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
		
		//read data representing the specified asset
		if (stoi(row[1]) == asset){
			
			rows.push_back(row);
		}
	}
	
}

//method to create training sequences and labels
//float* input -> pointer to float array with input data
//vector<float*>& sequences -> pointer to vector of output sequences
//vector<float*>& labels -> pointer to vector of output labels
//int seqSize -> num elements in a sequence
//int labelSize -> num elements in a label
//int inputSize -> num elements in the input data
void createSeqLabels(float* input, vector<float*>& sequences, vector<float*>& labels, int seqSize, int labelSize, int inputSize){
	float* temp;
	float* tempLabel;
	int temp_idx = 0;
	int temp_label= 0;
	cout << "Creating sequences and labels from " << inputSize << " data points" << endl;

	//only segment on data before the train/test split
	for (int index = 0; index < (inputSize - seqSize - labelSize); index+=labelSize){
		temp = new float[seqSize];
		temp_idx = 0;
		tempLabel = new float[labelSize];
		temp_label = 0; 
		
		//create a sequence 
		for (int sub = index; sub < index + seqSize; sub++){
			temp[temp_idx] = input[sub];
			temp_idx++; 
		}

		if (temp_idx == seqSize){
			sequences.push_back(temp);
		}	
	
		//create a label
		for (int sub = index + seqSize; sub < (index + seqSize + labelSize); sub++){
			tempLabel[temp_label] = input[sub];
			temp_label++; 
		}

		if (temp_label == labelSize){
			labels.push_back(tempLabel);
		}	
	}
	
}

//Perform min-max normalization
void normalize(float* inputFeature, float* outputFeature, int splitIdx){
	// setup cublas
	cublasStatus_t cbstatus;
	cublasHandle_t cbhandle;
	cublasCreate(&cbhandle);

	// allocate device memory for input/output
	float* cbInputFeature;
	float* devOutputFeature;
	cudaMalloc((void**)&cbInputFeature, splitIdx * sizeof(float));
	cudaMalloc((void**)&devOutputFeature, splitIdx * sizeof(float));
		
	//setup min/max variables
	int maxIndex, minIndex;
	float maxVal, minVal;
	float newMax = 1.0;
	float newMin = -1.0;

	//memcopy input to device
	cout << "Copying input feature data to device for CUBLAS" << endl;
	cudaMemcpy(cbInputFeature, inputFeature, splitIdx * sizeof(float), cudaMemcpyHostToDevice);
	
	//find max val
	cbstatus = cublasIsamax(cbhandle, splitIdx, cbInputFeature, 1, &maxIndex);
	if( cbstatus != CUBLAS_STATUS_SUCCESS){
		cerr << "CUBLAS error performing max" << endl;
		return;
	}
	
	//move max value to host
	cudaMemcpy(&maxVal, cbInputFeature+maxIndex-1, sizeof(float), cudaMemcpyDeviceToHost);
	maxVal = (maxVal >= 0) ? maxVal : -maxVal;
	cout << "Absolute max of input is " << maxVal << endl;
	
	//find min val
	cbstatus = cublasIsamin(cbhandle, splitIdx, cbInputFeature, 1, &minIndex);
	if( cbstatus != CUBLAS_STATUS_SUCCESS){
		cerr << "CUBLAS error performing max" << endl;
		return;
	}
	
	//move min value to host
	cudaMemcpy(&minVal, cbInputFeature+minIndex-1, sizeof(float), cudaMemcpyDeviceToHost);
	minVal = (minVal >= 0) ? minVal : -minVal;
	cout << "Absolute min of input is " << minVal << endl;

	// run normalization kernel
	int blockSize = 256;
	int numBlocks = (splitIdx + blockSize - 1) / blockSize;
	minmax<<<numBlocks, blockSize>>>(cbInputFeature, devOutputFeature, (newMax - newMin), (maxVal - minVal), newMin, minVal);
	
	// copy normalized data to host
	cudaMemcpy(outputFeature, devOutputFeature, splitIdx * sizeof(float), cudaMemcpyDeviceToHost);
	
	// print sample normalized data
	cout << "Normalization complete. Sample normalization data" << endl << endl;	
	for (int index =0; index < 10; index++){
		cout << "\tOriginal " << inputFeature[index] << " Normalized " << outputFeature[index] << endl;

	}
	cout << endl;
	
	// memory cleanup
	cudaFree(cbInputFeature);
	cudaFree(&devOutputFeature);
	cbstatus = cublasDestroy(cbhandle);

	if( cbstatus != CUBLAS_STATUS_SUCCESS){
		cerr << "CUBLAS shutdown error" << endl;
		return;
	}

}


//main function
int main(int argc, char const *argv[]) {
	int batch = 256; //batch size 
	int miniBatch =256;   	
	int inputSize; //number of input sequences
	int hiddenSize = 100; //hidden cell size
	int numLayers = 1; //number of network layers
	int seqSize = 1440; // number of data points in a sequence
	int labelSize = 60; // number of data points in a label
	//TODO expand code to train on multiple input features
	int features = 1; // number of features to train on
	float splitPercent = 0.8; // train-test data split
	int feature = 3; // index in input data of the feature to train on 
 	int numLinearLayers = 8;
	double paddingFill = 0.0;
 	
	//Ensure the correct number of arguments were passed
	if (argc < 3){
		cerr << "Usage: " << argv[0] << " <mode> <input>" << endl;
		exit(-1);
	}

	//parse train vs eval mode
	string procMode = argv[1];
	boost::to_upper(procMode);
 
	//get input file name
	string inputName = argv[2];	
	int asset = 1;
	
	vector<vector<string>> rows;
	vector<vector<string>> subset;

	//read input data from CSV
	readCsvFile(rows, argv[2], asset);
	cout << "Read " << rows.size() << " input rows" << endl;
	
	cout << "Created subset of data with asset " << asset << " containing " << rows.size() << " data points" << endl;	

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
	
	//perform normalization
	normalize(inputFeature, outputFeature, splitIdx);	

	//create training sequences and labels.
	vector<float*> sequences;
	vector<float*> labels;
	createSeqLabels(outputFeature, sequences, labels, seqSize, labelSize, splitIdx);
	
	cout << "Created " << sequences.size() << " sequences, " << labels.size() << " labels" << endl;

	//set input size to the number of sequences
	inputSize = sequences.size();	

	//setup input sequences on host

	//get device info
	int gpu;
	cudaSetDevice(0);
	struct cudaDeviceProp deviceProperties;
	cudaGetDevice(&gpu);
	cudaGetDeviceProperties(&deviceProperties, gpu);
	cout << "GPU device properties: " << deviceProperties.name << endl;

	//create CUDNN handle
	cudnnHandle_t cudnnHandle;
 	checkCUDNN(cudnnCreate(&cudnnHandle));
	cout << "Created CUDNN handle" << endl;

	//define tensor descriptors
	//tensor desciptors are default initialized to 0
	cudnnTensorDescriptor_t weightsDescriptor;
	cudnnTensorDescriptor_t biasDescriptor;
	//cudnnTensorDescriptor_t inputDescriptor;
    cudnnTensorDescriptor_t cLongTermDescriptor;
    cudnnTensorDescriptor_t hShortTermDescriptor;

	//create tensor descriptors
	checkCUDNN(cudnnCreateTensorDescriptor(&weightsDescriptor));
	checkCUDNN(cudnnCreateTensorDescriptor(&biasDescriptor));
	//checkCUDNN(cudnnCreateTensorDescriptor(&inputDescriptor));
	checkCUDNN(cudnnCreateTensorDescriptor(&cLongTermDescriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&hShortTermDescriptor));

	// initialize tensor descriptors
	//params: descriptor, data type, dimensions, size along dimensions, stride along dimensions 
	const int numDimensions = 3; //batch, features, data
    
	//hidden state desc init
	int hiddenDim[numDimensions] = {numLayers, miniBatch, hiddenSize};
	int hiddenStride[3] = {(hiddenDim[1] * hiddenDim[2]), hiddenDim[2], 1};
	checkCUDNN(cudnnSetTensorNdDescriptor(hShortTermDescriptor, CUDNN_DATA_FLOAT, numDimensions, hiddenDim, hiddenStride));
    checkCUDNN(cudnnSetTensorNdDescriptor(cLongTermDescriptor, CUDNN_DATA_FLOAT, numDimensions, hiddenDim, hiddenStride));
	cout << "Created hidden tensor descriptors" << endl;	
	
	//input desc init
	/*
	int inputDim[numDimensions] = {miniBatch, features, inputSize};
	checkCUDNN(cudnnSetTensorNdDescriptor(inputDescriptor, CUDA_DATA_FLOAT, numDimensions, inputDim, ));
	cout << "Created input tensor descriptor" << endl;	
	*/

	//define device arrays pointers
	void *deviceInput;
	void *deviceGradInput;
	void *deviceOutput;
	void *deviceGradOutput;
	void *deviceHiddenHX;	
	void *deviceHiddenHY;
	void *deviceHiddenCX;	
	void *deviceHiddenCY;	
	void *deviceHiddenGradHX;	
	void *deviceHiddenGradHY;	
	void *deviceHiddenGradCX;	
	void *deviceHiddenGradCY;	
	int *deviceSequenceLen;

	//allocate device input tensor
	int inputTensorSize = seqSize * miniBatch * inputSize * sizeof(CUDNN_DATA_FLOAT);
	cudaMalloc((void**)&deviceInput, inputTensorSize);
	cudaMalloc((void**)&deviceGradInput, inputTensorSize);
	cout << "Allocated input tensors " << endl;

	//allocate device output tensor
	int outputTensorSize = seqSize * miniBatch * hiddenSize * sizeof(CUDNN_DATA_FLOAT);
	cudaMalloc((void**)&deviceOutput, outputTensorSize);
	cudaMalloc((void**)&deviceGradOutput, outputTensorSize);
	cout << "Allocated output tensors " << endl;

	//allocate hidden tensors
	int hiddenTensorSize = numLayers * miniBatch * hiddenSize * sizeof(CUDNN_DATA_FLOAT);

	cudaMalloc((void**)&deviceHiddenHX, hiddenTensorSize);
	cudaMalloc((void**)&deviceHiddenCX, hiddenTensorSize);
	cudaMalloc((void**)&deviceHiddenHY, hiddenTensorSize);
	cudaMalloc((void**)&deviceHiddenCY, hiddenTensorSize);
	cudaMalloc((void**)&deviceHiddenGradHX, hiddenTensorSize);
	cudaMalloc((void**)&deviceHiddenGradCX, hiddenTensorSize);
	cudaMalloc((void**)&deviceHiddenGradHY, hiddenTensorSize);
	cudaMalloc((void**)&deviceHiddenGradCY, hiddenTensorSize);
	cout << "Allocated output tensors " << endl;

	int* hostSequenceLen = (int*)malloc(miniBatch*sizeof(int));
	for (int i = 0; i < miniBatch; i++){
		hostSequenceLen[i] = seqSize;

	}
	//allocate device input sequence		
	cudaMalloc((void **)&deviceSequenceLen, miniBatch * sizeof(int));
	cudaMemcpy(deviceSequenceLen, hostSequenceLen, miniBatch * sizeof(int), cudaMemcpyHostToDevice);	
	cout << "Allocated input sequence " << endl;

	// Create RNN Data descriptors
	cudnnRNNDataDescriptor_t xRNNDescriptor;
	cudnnRNNDataDescriptor_t yRNNDescriptor;
    checkCUDNN(cudnnCreateRNNDataDescriptor(&xRNNDescriptor));
    checkCUDNN(cudnnCreateRNNDataDescriptor(&yRNNDescriptor));

	// Initialize RNN Data descriptors
    checkCUDNN(cudnnSetRNNDataDescriptor(xRNNDescriptor,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                         seqSize,
                                         miniBatch,
                                         inputSize,
                                         hostSequenceLen,
                                         &paddingFill));

    checkCUDNN(cudnnSetRNNDataDescriptor(yRNNDescriptor,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                         seqSize,
                                         miniBatch,
                                         hiddenSize,
                                         hostSequenceLen,
                                         &paddingFill));

	cout << "Created RNN data descriptors" << endl;

	// Set up the dropout descriptor (needed for the RNN descriptor)
    unsigned long long seed = 1337ull;
	float dropout = 0;
	cudnnDropoutDescriptor_t dropoutDescriptor;
	size_t stateSize;
    void   *states;
	checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDescriptor));
    checkCUDNN(cudnnDropoutGetStatesSize(cudnnHandle, &stateSize));
   	cudaMalloc(&states, stateSize);
    checkCUDNN(cudnnSetDropoutDescriptor(dropoutDescriptor,
                                            cudnnHandle,
                                            dropout,
                                            states,
                                            stateSize,
                                            seed));




    //Create RNN Descriptor
	cudnnRNNDescriptor_t RNNDescriptor;
	checkCUDNN(cudnnCreateRNNDescriptor(&RNNDescriptor));

	//initialize RNN Descriptor
	//no droupout in single layer network
    checkCUDNN(cudnnSetRNNDescriptor_v8(RNNDescriptor,
                                     CUDNN_RNN_ALGO_PERSIST_DYNAMIC,
                                     CUDNN_LSTM,
                                     CUDNN_RNN_NO_BIAS,
                                     CUDNN_UNIDIRECTIONAL,
                                     CUDNN_LINEAR_INPUT,
                                     CUDNN_DATA_FLOAT,
                                     CUDNN_DATA_FLOAT,
                                     CUDNN_DEFAULT_MATH,
                                     inputSize,
                                     hiddenSize,
                                     labelSize,
                                     numLayers,
                                     dropoutDescriptor,
                                        0));
	cout << "Created RNN descriptor" << endl;

    // Set up weights and bias parameters
	
	size_t weightSpaceSize;
	void *weightSpace;
	void *gradWeightSpace;
	
    checkCUDNN(cudnnGetRNNWeightSpaceSize(cudnnHandle,RNNDescriptor, &weightSpaceSize));
	cudaMalloc((void **)&weightSpace, weightSpaceSize);
    cudaMalloc((void **)&gradWeightSpace, weightSpaceSize);
	cout << "Allocated weight space" << endl;

	//Initialize working space and reserved space
	void *workSpace;
    void *reserveSpace;

    size_t workSpaceSize;
    size_t reserveSpaceSize;	

    checkCUDNN(cudnnGetRNNTempSpaceSizes(cudnnHandle,
                                         RNNDescriptor,
                                         CUDNN_FWD_MODE_TRAINING,
                                         xRNNDescriptor,
                                         &workSpaceSize,
                                         &reserveSpaceSize));

    cudaMalloc((void **)&workSpace, workSpaceSize);
    cudaMalloc((void **)&reserveSpace, reserveSpaceSize);
	cudaMemset(gradWeightSpace, 0, weightSpaceSize);	
	
	cout << "Weight space size in MiB: " << weightSpaceSize/1024.0/1024.0 << endl;
	cout << "Work space size in MiB: " << workSpaceSize/1024.0/1024.0 << endl;
	cout << "Reserve space size in MiB: " << reserveSpaceSize/1024.0/1024.0 << endl;

	// Create a dynamic persistent RNN plan
    checkCUDNN(cudnnBuildRNNDynamic(cudnnHandle, RNNDescriptor, batch));
	cout << "Built dynamic persistent RNN plan" << endl;
	
	//Training!
	
	cudaDeviceSynchronize();
	//Forward pass
	checkCUDNN(cudnnRNNForward(cudnnHandle,
                               RNNDescriptor,
                               CUDNN_FWD_MODE_TRAINING,
                               deviceSequenceLen,
                               xRNNDescriptor,
                               deviceInput,
                               yRNNDescriptor,
                               deviceOutput,
                               hShortTermDescriptor,
                               deviceHiddenHX,
                               deviceHiddenHY,
                               cLongTermDescriptor,
                               deviceHiddenCX,
                               deviceHiddenCY,
                               weightSpaceSize,
                               weightSpace,
                               workSpaceSize,
                               workSpace,
                               reserveSpaceSize,
                               reserveSpace));

	//Backward pass on data
	checkCUDNN(cudnnRNNBackwardData_v8(cudnnHandle,
                                       RNNDescriptor,
                                       deviceSequenceLen,
                                       yRNNDescriptor,
                                       deviceOutput,
                                       deviceGradOutput,
                                       xRNNDescriptor,
                                       deviceGradInput,
                                       hShortTermDescriptor,
                                       deviceHiddenHX,
                                       deviceHiddenGradHY,
                                       deviceHiddenGradHX,
                                       cLongTermDescriptor,
                                       deviceHiddenCX,
                                       deviceHiddenGradCY,
                                       deviceHiddenGradCX,
                                       weightSpaceSize,
                                       weightSpace,
                                       workSpaceSize,
                                       workSpace,
                                       reserveSpaceSize,
                                       reserveSpace));

	// Backward pass for weights
	checkCUDNN(cudnnRNNBackwardWeights_v8(cudnnHandle,
                                          RNNDescriptor,
                                          CUDNN_WGRAD_MODE_ADD,
                                          deviceSequenceLen,
                                          xRNNDescriptor,
                                          deviceInput,
                                          hShortTermDescriptor,
                                          deviceHiddenHX,
                                          yRNNDescriptor,
                                          deviceOutput,
                                          weightSpaceSize,
                                          gradWeightSpace,
                                          workSpaceSize,
                                          workSpace,
                                          reserveSpaceSize,
                                          reserveSpace));

	cudaDeviceSynchronize();
	
	//memory cleanup	
	cout << "Done! cleaning up" << endl;
	free(hostSequenceLen);
	cudaFree(deviceSequenceLen);
	cudaFree(deviceInput);
	cudaFree(deviceGradInput);
	cudaFree(deviceOutput);
	cudaFree(deviceGradOutput);
	cudaFree(deviceHiddenHX);
	cudaFree(deviceHiddenHY);
	cudaFree(deviceHiddenCX);
	cudaFree(deviceHiddenCY);
	cudaFree(deviceHiddenGradHX);
	cudaFree(deviceHiddenGradHY);
	cudaFree(deviceHiddenGradCX);
	cudaFree(deviceHiddenGradCY);
	cudaFree(workSpace);
	cudaFree(reserveSpace);
	cudaFree(weightSpace);
	cudaFree(gradWeightSpace);
	cudaFree(deviceSequenceLen);

	checkCUDNN(cudnnDestroyRNNDataDescriptor(xRNNDescriptor));
	checkCUDNN(cudnnDestroyRNNDataDescriptor(yRNNDescriptor));
	
	checkCUDNN(cudnnDestroyTensorDescriptor(biasDescriptor));
	checkCUDNN(cudnnDestroyTensorDescriptor(weightsDescriptor));
	//checkCUDNN(cudnnDestroyTensorDescriptor(inputDescriptor));
	checkCUDNN(cudnnDestroyTensorDescriptor(hShortTermDescriptor));
	checkCUDNN(cudnnDestroyTensorDescriptor(cLongTermDescriptor));
	checkCUDNN(cudnnDestroyDropoutDescriptor(dropoutDescriptor));	
    checkCUDNN(cudnnDestroyRNNDescriptor(RNNDescriptor));
    cudnnDestroy(cudnnHandle);
	
}
