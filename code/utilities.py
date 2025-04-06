import os
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import subprocess
from models import *


class pepDataset(Dataset):
    def __init__(self, peptide_tokens, affinity_label):
        self.peptide_tokens = peptide_tokens
        self.affinity_label = affinity_label

    def __len__(self):
        return len(self.affinity_label)

    def __getitem__(self, idx):
        sequence = self.peptide_tokens[idx].to(torch.int64)
        label = torch.tensor(self.affinity_label[idx], dtype=torch.float32)
        return sequence, label


def load_tokenizer():
    folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    tok2idx_path = os.path.join(folder_path, 'data', 'tok2idx.pkl')
    with open(tok2idx_path, 'rb') as f:
        tok2idx = pickle.load(f)
    return tok2idx


def decode_generated_peptides(fake_peptides):
    tok2idx = load_tokenizer()
    predicted_indices = fake_peptides.argmax(dim=-1) + 2
    idx2tok = {idx: tok for tok, idx in tok2idx.items()}
    decoded_tokens = [''.join([idx2tok[idx.item()] for idx in seq]) for seq in predicted_indices]
    return decoded_tokens


def test_generation_diversity(fake_peptides):
    decoded_tokens = decode_generated_peptides(fake_peptides)
    return len(set(decoded_tokens)) / len(decoded_tokens)


def get_free_gpu():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE
    )
    free_memory = result.stdout.decode('utf-8').strip().split('\n')
    free_memory = [int(x) for x in free_memory]

    best_gpu = free_memory.index(max(free_memory))
    return best_gpu


def select_device():
    if torch.cuda.is_available():
        torch.cuda.init()
        gpu_index = get_free_gpu()
        device = torch.device(f'cuda:{gpu_index}')
        print(f"Using GPU {gpu_index} with the most free memory.")
    else:
        device = torch.device('cpu')
        print("No GPU available. Using CPU.")
    return device


def predict_affinity_score(peptides_list, plastic_name, device, batch_size = 1000):

    folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # tokenize peptide sequence
    tok2idx = load_tokenizer()
    peptide_tokens = []
    for peptide in peptides_list:
        tokens = [tok2idx[acid] for acid in peptide]
        peptide_tokens.append(tokens)
    peptide_tokens = torch.tensor(peptide_tokens)
    peptide_onehot = F.one_hot(peptide_tokens, num_classes=20).float()

    # load pre-trained predictor model
    esm_model = modifyESM2()
    esm_model.to(device)
    esm_model_path = os.path.join(folder_path, 'data', plastic_name+'_esm_model.pth')
    esm_model.load_state_dict(torch.load(esm_model_path, map_location=device, weights_only=True))

    predictor = predictorNet()
    predictor.to(device)
    predictor_path = os.path.join(folder_path, 'data', plastic_name+'_predictor.pth')
    predictor.load_state_dict(torch.load(predictor_path, map_location=device, weights_only=True))

    esm_model.eval()
    predictor.eval() 

    predict_score_list = []

    for i in range(0, len(peptides_list), batch_size):
        endidx = min(i + batch_size, len(peptides_list))
        batch_onehot = peptide_onehot[i:endidx]
        batch_onehot = batch_onehot.to(device)
        predict_score_list.extend(predictor(esm_model(batch_onehot)).squeeze(-1).tolist()) 

    return predict_score_list




