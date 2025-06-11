import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
import argparse
import random
import numpy as np
import pandas as pd
from utilities import *


parser = argparse.ArgumentParser()
parser.add_argument('-p', "--plastic_name", type = str, default = 'PE')
parser.add_argument('-bz', "--batch_size", type = int, default = 64)
parser.add_argument('-lr', "--learning_rate", type = float, default = 1e-4)
parser.add_argument('-epch', "--num_epochs", type = int, default = 20)
args = parser.parse_args()


plastic_name = args.plastic_name
running_mode = args.running_mode
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.num_epochs


random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(random_seed)
	torch.cuda.manual_seed_all(random_seed)

# select device
device = select_device()

# get script and folder name
script_name = os.path.splitext(os.path.basename(__file__))[0]

folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load tokenizer
tok2idx = load_tokenizer()
print("Tokenizer Loaded")

# load peptide data
data_folder_path = os.path.join(folder_path, 'data')
df = pd.read_csv(os.path.join(data_folder_path, plastic_name + '_clean.csv'))
print("Peptide Data Loaded # ", df.shape[0])

peptide_sequence = df.iloc[:, 0].to_list()
affinity_label = df.iloc[:, 1].to_list()

# tokenize peptide sequence
peptide_tokens = []
for peptide in peptide_sequence:
	tokens = [tok2idx[acid] for acid in peptide]
	peptide_tokens.append(tokens)
peptide_tokens = torch.tensor(peptide_tokens)

dataset = pepDataset(peptide_tokens, affinity_label)
total_size = len(dataset)

test_size = int(0.2 * total_size)
remaining_size = total_size - test_size
train_size = int(0.6 * total_size)
val_size = remaining_size - train_size
dataset_remain, test_dataset = random_split(dataset, [remaining_size, test_size])

test_loader = DataLoader(
	test_dataset,
	batch_size=batch_size,
	shuffle=False,
	drop_last=False
)
print(f"Total samples: {total_size}, Remaining for train/val: {remaining_size}, Test: {test_size}")

# load pre-trained predictor model
esm_model = modifyESM2()
esm_model.to(device)
esm_model_path = os.path.join(folder_path, 'data', 'predictor', 'pretrained_ESM6_modified.pth')
esm_model.load_state_dict(torch.load(esm_model_path, map_location=device, weights_only=True))
print('Loaded pretrained ESM-2 Model')

predictor = predictorNet()
predictor.to(device)

esm_model_optimizer = torch.optim.Adam(esm_model.parameters(), lr=learning_rate)
esm_model_scheduler = StepLR(esm_model_optimizer, step_size=4, gamma=0.5)

predictor_optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
predictor_scheduler = StepLR(predictor_optimizer, step_size=4, gamma=0.5)

criterion = nn.MSELoss()


for epoch in range(num_epochs):

	train_dataset, val_dataset = random_split(
		dataset_remain,
		[train_size, val_size]
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		drop_last=False
	)

	running_loss = 0.0

	esm_model.train()
	predictor.train()
	for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):

		batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
		batch_onehot = F.one_hot(batch_data, num_classes=20).float()
		outputs = predictor(esm_model(batch_onehot))
		loss = criterion(outputs.squeeze(), batch_labels)  

		esm_model_optimizer.zero_grad() 
		predictor_optimizer.zero_grad()
		loss.backward() 
		esm_model_optimizer.step() 
		predictor_optimizer.step()

		running_loss += loss.item()

		if (batch_idx + 1) % 200 == 0:
			print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

	print(f"Epoch [{epoch + 1}/{num_epochs}] Training Average Loss: {running_loss / len(train_loader):.4f}")

	# adjust learning rate
	esm_model_scheduler.step()
	predictor_scheduler.step()
	current_lr = esm_model_optimizer.param_groups[0]['lr']
	print(f"Epoch {epoch + 1}: Learning Rate = {current_lr}")


	# validate model
	predictor.eval() 
	esm_model.eval()
	val_loss = 0.0
	with torch.no_grad(): 
		for batch_idx, (batch_data, batch_labels) in enumerate(val_loader):

			batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
			batch_onehot = F.one_hot(batch_data, num_classes=20).float()
			outputs = predictor(esm_model(batch_onehot))
			val_loss += criterion(outputs.squeeze(), batch_labels).item()

	print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss / len(val_loader):.4f}")


	# test model
	test_loss = 0.0
	with torch.no_grad(): 
		for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):

			batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
			
			batch_onehot = F.one_hot(batch_data, num_classes=20).float()
			outputs = predictor(esm_model(batch_onehot))
			test_loss += criterion(outputs.squeeze(), batch_labels).item()

	print(f"Epoch [{epoch + 1}/{num_epochs}] Test Loss: {test_loss / len(test_loader):.4f}")


	# Save model parameters
	predictor_save_path = os.path.join(folder_path, 'data', 'predictor', plastic_name + '_predictor_'+'epch_'+ "{:02}".format(epoch + 1) +'.pth')
	torch.save(predictor.state_dict(), predictor_save_path)
	print('Saved ' + predictor_save_path)

	esmModel_save_path = os.path.join(folder_path, 'data', 'predictor', plastic_name + '_esm_model_'+ 'epch_'+ "{:02}".format(epoch + 1) +'.pth')
	torch.save(esm_model.state_dict(), esmModel_save_path)
	print('Saved ' + esmModel_save_path)








