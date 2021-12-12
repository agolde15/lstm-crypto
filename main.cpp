#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <cudnn.h>
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

void readCsvFile(vector<vector<string>>& rows, std::string fileName){
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
		if (stoi(row[1]) == 0){
			rows.push_back(row);
		} 
	}
	
}


int main(int argc, char const *argv[]) {
 	if (argc < 3){
		cerr << "Usage: " << argv[0] << " <mode> <input>" << endl;
		exit(-1);
	}
	string procMode = argv[1];
	boost::to_upper(procMode);
 
	string inputName = argv[2];	

	if (procMode.compare("TRAIN") == 0) {
		vector<vector<string>> rows;
		readCsvFile(rows, argv[2]);
		cout << "Read " << rows.size() << " input rows" << endl;
		for (int index = 0; index < 10; index++){
			std::cout << rows[index].front() << std::endl;	

		 }
 	} else if (procMode.compare("INFER") == 0){
		cout << "Inference coming soon" << endl;
		
	} 

	 cudnnHandle_t cudnnHandle;
	 cudnnStatus_t status;
 	 checkCUDNN(cudnnCreate(&cudnnHandle));
	 cudnnRNNDataDescriptor_t RNNDataDesc;
     checkCUDNN(cudnnCreateRNNDataDescriptor(&RNNDataDesc));
	 //checkCUDNN(cudnnRNNForward(&cudnnHandle, &RNNDataDesc, CUDNN_FWD_MODE_TRAINING));
     auto deviceMemoryAvailable  = getDeviceMemory();
	 cout << deviceMemoryAvailable << endl;
   	 cudnnDestroy(cudnnHandle);

}
