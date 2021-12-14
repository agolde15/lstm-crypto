#Welcome to the Cryptocurrency Predictions Machine Learning project!
This project is intended to train a LSTM neural network using the training data provided from the
G-Research Crypto Forecasting competition. The dataset is available here:
https://www.kaggle.com/c/g-research-crypto-forecasting/data

The dataset, specifically the train.csv file is NOT included in this repository due to size constraints.
The dataset should be downloaded before using this project

There are two implementations for the LSTM model in this project: PyTorch and CUDA.

### PyTorch Implementation:
The PyTorch implementation uses PyTorch to train and run inference with the LSTM model.
The requirements.txt file lists the python requirements for this project.

To run training:
python3 lstm.py --file data/train.csv --train --verbose --asset <asset idx> --checkpoint <file name> --batch <batch size>

The asset idx should be an integer between 0 and 13, inclusive, representing which asset should be selected.
The checkpoint file name is a base name for where the code will output the checkpoints to. 
Batch size should be an integer multiple of 2 and may need to be reduced depending on the GPU memory size.

To run inference:
python3 lstm.py --file data/train.csv --verbose --asset 1 --checkpoint <file name> --batch 1 --output <num samples>


The asset idx should be an integer between 0 and 13, inclusive, representing which asset should be selected.
The checkpoint file name is a the filename the model state should be read from. 
The output parameter lists how many samples the model should predict.

Please run python3 lstm.py --help to see additional parameter descriptions.

### Cuda Implementation
Required Dependencies:
 * boost 1.77.0
 * Cuda 11.5
 * cuDNN
 * cuBlas
The implementation code is located in main.cu.
Once the dependencies are installed on a system with a NVIDIA GPU, then the user can run "make" to build the source code.
To run training:
./main train data/train.csv
