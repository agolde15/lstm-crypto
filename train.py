import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import genfromtxt
'''
Params:
	input_size – The number of expected features in the input x
    hidden_size – The number of features in the hidden state h
    num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
    bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False
    dropout – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
    bidirectional – If True, becomes a bidirectional LSTM. Default: False
    proj_size – If > 0, will use LSTM with projections of corresponding size. Default: 0
'''

def main():
	parser = optparse.OptionParser()
	parser.add_option("-f", "--file", dest="filename", help="path to input file")
	parser.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="print status messages to stdout")
	parser.add_option("-t", "--train", action="store_true", dest="train", default=False, help="train the network on file specified")
	parser.add_option("-a", "--asset", dest="asset",type=int, help="index representing asset number")
	parser.add_option("-c", "--checkpoint", dest="checkpoint", help="filename of checkpoint to load for eval")
	parser.add_option("-w", "--window", dest="window",type=int, default=1440, help="number of timesteps in sequence")
	parser.add_option("-l", "--label", dest="label",type=int, default=60, help="number of timesteps in label/target")
	parser.add_option("-e", "--epochs", dest="epochs",type=int, default=1, help="number of training epochs")
	parser.add_option("-i", "--hidden", dest="hidden",type=int, default=100, help="number of hidden cells")
	parser.add_option("-y", "--layers", dest="layers",type=int, default=1, help="number of layers")
	parser.add_option("-b", "--batch", dest="batch",type=int, default=1048, help="batch size")
	parser.add_option("-p", "--percent", dest="percent",type=float, default=0.8, help="train/test split percentage")
	parser.add_option("-r", "--lr", dest="lr",type=float, default=0.001, help="learning rate")
	parser.add_option("-o", "--output", dest="output",type=int, default=60, help="number of timesteps to predict in eval. Must be >= label size")
	(opts, args) = parser.parse_args()
	
	window = opts.window
	label = opts.label
	batch = opts.batch
	layers = opts.layers
	hidden_cells_num = opts.hidden

	if not opts.filename:
		parser.error("Input file is required")

	if not opts.checkpoint:
		parser.error("Checkpoint file is required (provide existing file for EVAL loading and ouput filename for TRAIN save")
		
	#check that input file is CSV
	input_extension = os.path.splitext(opts.filename)[1]
	if input_extension != '.csv':
		print('Error: in training mode, input file must be CSV.')
		sys.exit(-1)

	df = pd.read_csv(opts.filename, delimiter=',', parse_dates=[0], infer_datetime_format=True)
	asset_df = df[df['Asset_ID'] == opts.asset]
	data = asset_df.to_numpy()
	#plt.figure()
	#df[['timestamp', 'Asset_ID', 'Open']].plot()
	#plt.show()
	#print(df)
	#split train/test data
	len_data = np.shape(data)[0]
	test_percent = opts.percent
	train_test_split_idx = int(test_percent * len_data)
	if opts.verbose: print("Split index: ", train_test_split_idx)
	test_data = data[train_test_split_idx:, 3]

	#normalize training data
	scaler = MinMaxScaler(feature_range=(-1, 1))
	normalized_data = scaler.fit_transform(data[:train_test_split_idx, 3].reshape(-1, 1))
	if opts.verbose: print("original data sample", data[0:20, 3])
	if opts.verbose: print("normalizaed data sample", normalized_data[0:20, :])
	train_data = torch.FloatTensor(normalized_data)
	
	if opts.verbose: print("Input data shape:", np.shape(data))
	if opts.train:

		# create input sequences and labels 
		subsets = []
		label_vals = []
		# increment by <label> samples so that labels do not overlap
		for i in range(0, len_data - window- label - 1, label):
			subset = train_data[i:i+window, :]
			
			label_val = train_data[i + window:i+window+label]
			subsets.append(subset)
			label_vals.append(label_val)

		# divide input sequences and labels into batches
		batches = []
		for i in range(len(subsets) % batch):
			temp_b = torch.stack(subsets[i * batch: i*batch + batch])
			temp_l =torch.stack(label_vals[i * batch: i*batch + batch])
			batches.append((temp_b, temp_l))
		if opts.verbose: print("Sample batch dims: ", batches[0][0].shape)
		
		#initialize cuda device usage
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		if opts.verbose: print("Using device", torch.cuda.get_device_name(0), device)

		#create model
		model = LSTMCrypto(layers, hidden_cells_num, label, batch, layers, batch_first=True)
		if opts.verbose: print("Model: ", model)
		#move model to GPU/device
		model.to(device)
		model.train()

		#loss function setup
		loss_func = nn.MSELoss()
		if opts.verbose: print("Using MSE loss function")

		#learning rate setup
		learning_rate = opts.lr
		if opts.verbose: print("Learning rate: ", learning_rate)
		
		#optimizer setup
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
		if opts.verbose: print("Using Adam optimizer")
		
		#set number of training epochs
		epochs = opts.epochs
		
		if opts.verbose: print(f"Beginning training for {epochs} epochs")
		for i in range(0, epochs):
			for sequence, labels in batches:
				optimizer.zero_grad()
				model.hidden_cell = (torch.zeros(model.layers, model.batch, model.hidden_dim).cuda(),
									 torch.zeros(model.layers, model.batch, model.hidden_dim).cuda())
				pred = model(sequence.cuda())
				
				loss = loss_func(pred, labels.cuda())
				loss.backward()
				optimizer.step()	
			if opts.verbose:
				print(f'Epoch: {i} loss: {loss.item():10.8f}')
		
			torch.save(model.state_dict(), "epoch-" + str(i) + opts.checkpoint)		
		print(f'Epoch: {i} loss: {loss.item():10.8f}')
	
	#evaluation mode
	if not opts.train:
		#initialize cuda device usage
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		if opts.verbose: print("Using device", torch.cuda.get_device_name(0), device)

		model = LSTMCrypto(layers, hidden_cells_num, label, batch, layers, batch_first=True)
		model.load_state_dict(torch.load(opts.checkpoint))
		if opts.verbose: print("Model: ", model)
		#move model to GPU/device
		model.to(device)
		model.eval()
		
		eval_seed = normalized_data[-window:].tolist()
		eval_sequence = torch.FloatTensor(eval_seed[-window:]).unsqueeze(0).cuda()
		print("Len eval", len(eval_seed))
		for i in range(opts.output // label):
			print(eval_sequence.shape)
			eval_sequence = eval_sequence[:, -window:]
			with torch.no_grad():
				model.hidden_cell = (torch.zeros(model.layers, model.batch, model.hidden_dim).cuda(),
									 torch.zeros(model.layers, model.batch, model.hidden_dim).cuda())
				out = model(eval_sequence.cuda())
				eval_sequence = torch.cat((eval_sequence, out), dim=1)
				print(eval_sequence.shape)
		
		#device to host, remove batch dimension
		eval_sequence = eval_sequence.cpu().squeeze(0).detach().numpy() 
		if opts.verbose: print("Eval out", eval_sequence, eval_sequence.shape)
		
		unnormalized_preds = scaler.inverse_transform(eval_sequence.reshape(-1,1))
		if opts.verbose: print("Unnormalized preds", unnormalized_preds)
		print(unnormalized_preds.shape)
		x = np.arange(window, window + label,1)
		plt.title('Actual vs Preds')
		plt.grid(True)
		plt.autoscale(axis='x', tight=True)
		plt.plot(data[-window:,3])
		plt.plot(x, unnormalized_preds[-label:, :])
		plt.xlabel("Sample #")
		plt.ylabel("Price")
		plt.show() 
	
		#torch.manual_seed(1)

class LSTMCrypto(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, batch, layers=1, batch_first=False):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.batch = batch
		self.layers = layers
		self.batch_first = batch_first
		self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=batch_first)
		self.linear = nn.Linear(hidden_dim, output_dim)
		#this tracks previous cell state
		self.hidden_cell = (torch.zeros(layers,batch,self.hidden_dim), 
							torch.zeros(layers,batch,self.hidden_dim))
	def forward(self, input_sequence):
		lstm_out, self.hidden_cell = self.lstm(input_sequence, self.hidden_cell)
		preds = self.linear(lstm_out[:, -1, :]) #.view(len(input_sequence), -1))
		return preds.unsqueeze(2)
		


rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
print(output, hn, cn)
if __name__ == "__main__":
	main()
