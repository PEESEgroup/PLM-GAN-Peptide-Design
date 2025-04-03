import os
import random
import numpy as np
import pandas as pd
import torch
import argparse
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models import modifyESM2, generatorNet, predictorNet, criticNet
from utilities import *


def score_weight_scheduler(epoch, weight_rate):
    return max(0, epoch / weight_rate - 1.0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--random_seed", type = int, default = 0)
    parser.add_argument('-w', "--weight_rate", type = float, default = 40.0)
    parser.add_argument('-id', "--script_id", type = str, default = '00')
    args = parser.parse_args()

    print('script_id =', args.script_id)
    print('random_seed =', args.random_seed)
    print('weight_rate =', args.weight_rate)
    print('------------------------------------')

    # hyper-parameters
    latent_dim = 100        # dim of z
    num_epochs = 200
    batch_size = 64
    critic_iters = 5        # Number of critic updates per generator update
    learning_rate = 1e-5

    gumbel_tau = 1
    lambda_gp = 10          # Gradient penalty coefficient
    hydrophobic_penalty = 10

    disp_freq = 200         # display frequency
    model_save_freq = 5     # model save frequency for every epoch

    plastic_list = ['PE', 'PP', 'PET']

    random_seed = args.random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    # select device
    device = select_device()

    # get script and folder name
    script_name = os.path.splitext(os.path.basename(__file__))[0] + args.script_id

    if os.name == 'nt': # windows
        folder_path = os.path.dirname(os.path.abspath(__file__))
    else:
        folder_path = "./"

    save_model_dir = os.path.join(folder_path, script_name)
    if not os.path.exists(save_model_dir): 
        os.makedirs(save_model_dir)

    # load tokenizer
    tok2idx = load_tokenizer()
    print("Tokenizer Loaded")

    # load peptide data
    data_folder_path = os.path.join(folder_path, "PepBD")
    df_data = pd.read_csv(os.path.join(data_folder_path,'collected_peptides.csv'))

    if os.name == 'nt': # windows
        df_data = df_data.sample(n=1000, random_state=random_seed) 
    print("Peptide Data Loaded # ", df_data.shape[0])

    peptide_sequence = df_data.iloc[:, 0].to_list()
    affinity_label = df_data.iloc[:, 1].to_list()

    # tokenize peptide sequence
    peptide_tokens = []
    for peptide in peptide_sequence:
        tokens = [tok2idx[acid] for acid in peptide]
        peptide_tokens.append(tokens)
    peptide_tokens = torch.tensor(peptide_tokens)

    dataset = pepDataset(peptide_tokens, affinity_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print("Dataset and Dataloader Constructed")

    # load pre-trained predictor model
    esm_model_list = []
    predictor_list = []

    for plastic in plastic_list:
        esm_model = modifyESM2()
        esm_model.to(device)
        esm_model_path = os.path.join(folder_path, 'model_save', plastic + '_esm_model.pth')
        esm_model.load_state_dict(torch.load(esm_model_path, map_location=device, weights_only=True))
        esm_model_list.append(esm_model)
    print('Loaded esm_model Model')

    for plastic in plastic_list:
        predictor = predictorNet()
        predictor.to(device)
        predictor_path = os.path.join(folder_path, 'model_save', plastic + '_predictor.pth')
        predictor.load_state_dict(torch.load(predictor_path, map_location=device, weights_only=True))
        predictor_list.append(predictor)
    print('Loaded predictor Model')

    # creat critic and generator
    generator = generatorNet(latent_dim=latent_dim)
    generator.to(device)

    critic = criticNet()
    critic.to(device)


    # Optimizers
    optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    optimizer_critic = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    esm_model.eval()
    predictor.eval() 
    critic.train()
    generator.train()

    # Initialize the DataFrame to store generated peptides and scores
    col_names = ['Epoch', 'Sequence'] + [p +' Score' for p in plastic_list]
    peptides_df = pd.DataFrame(columns=col_names)

    for epoch in range(num_epochs):
        for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):

            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # Train the critic
            optimizer_critic.zero_grad()
            
            # real peptide data
            real_peptides = F.one_hot(batch_data-2, num_classes=18).float() * 2 - 1.0 # [-1, 1]
            
            # Generate fake peptides using the generator
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_peptides = generator(z)
            
            # Get critic scores for real and fake samples
            real_output = critic(real_peptides)
            fake_output = critic(torch.tanh(fake_peptides))
            
            # Wasserstein loss for the critic
            c_loss = wasserstein_loss(fake_output, real_output)
            
            # gradient penalty
            gp = gradient_penalty(critic, real_peptides, fake_peptides)
            
            # Final critic loss with gradient penalty
            c_loss_total = c_loss + lambda_gp * gp
            
            c_loss_total.backward()
            optimizer_critic.step()

            # Train the generator (after critic_iters updates)
            if (batch_idx+1) % critic_iters == 0:
                
                optimizer_generator.zero_grad()
                
                # Generate fake peptides
                z = torch.randn(batch_size, latent_dim, device=device)

                # Calculate generation diversity to avoid model collapse
                fake_peptides_logit = generator(z)
                diversity = test_generation_diversity(fake_peptides_logit)

                # Get critic score for fake peptides
                fake_output = critic(torch.tanh(fake_peptides_logit))
                wasser_loss = wasserstein_loss(fake_output)

                fake_peptides_logit = F.gumbel_softmax(fake_peptides_logit, tau=gumbel_tau, hard=True)
                fake_peptides = F.pad(fake_peptides_logit, (2, 0)) # pad first two logits for <cls> <eos> in esm_model embedding

                # Hydrophobic penalty for amino acid W
                hydrophobic_loss = torch.relu(fake_peptides[:, :, tok2idx['W']].sum(-1) - 3.0).sum()

                # Calculate average score for the generated peptides
                fake_score_list = []
                detail_score_list = [] # [塑料种类] * [batch_size]
                for esm_model, predictor in zip(esm_model_list, predictor_list):
                    predicted_affinity = predictor(esm_model(fake_peptides))
                    detail_score_list.append(predicted_affinity)
                    fake_score_list.append(torch.mean(predicted_affinity))
                avg_socre = torch.mean(torch.stack(fake_score_list, dim=0), dim=0)

                # Compute Wasserstein loss + affinity score for the generator
                score_weight = score_weight_scheduler(epoch, args.weight_rate)
                g_loss = wasser_loss + hydrophobic_penalty * hydrophobic_loss + score_weight * avg_socre
                
                g_loss.backward()
                optimizer_generator.step()

                # Save generated peptides and scores to the DataFrame
                if score_weight > 0.5:
                    decode_peptides = decode_generated_peptides(fake_peptides_logit)
                    for idx in range(batch_size):
                        row_list = [epoch + 1] + [decode_peptides[idx]] + [detail_score_list[p][idx].item() for p in range(len(plastic_list))]
                        peptides_df.loc[len(peptides_df)] = row_list
                    

            # Display
            if (batch_idx + 1) % disp_freq == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}] Critic: {c_loss_total.item():.2f}  Wasserstein: {wasser_loss.item():.2f} Diversity: {diversity:.4f}", end = ' ')
                for plastic, fake_score in zip(plastic_list, fake_score_list):
                    print(f"{plastic}: {fake_score.item():.2f} ", end = ' ')
                print(f'Score: {avg_socre.item():.2f}')


        # Save generator， critic models and generated peptides
        if (epoch+1) % model_save_freq == 0 and score_weight > 0.5:
            
            generator_save_path = os.path.join(folder_path, script_name, 'generator_'+ 'epch_'+ "{:03}".format(epoch + 1) +'.pth')
            torch.save(generator.state_dict(), generator_save_path)
            print('Saved ' + generator_save_path)

            critic_save_path = os.path.join(folder_path, script_name, 'critic_'+ 'epch_'+ "{:03}".format(epoch + 1) +'.pth')
            torch.save(critic.state_dict(), critic_save_path)
            print('Saved ' + critic_save_path)

            peptides_df = peptides_df.drop_duplicates(subset="Sequence", keep="last")
            peptides_df = peptides_df[peptides_df['Sequence'].apply(lambda x: x.count('W') <= 3)] # Constrain hydrophobic amino acid W<=3
            save_df = peptides_df.copy()
            col_list = [p +' Score' for p in plastic_list]
            save_df["Average Score"] = save_df[col_list].mean(axis=1)
            save_df = save_df.sort_values(by='Average Score', ascending=True)
            csv_save_path = os.path.join(folder_path, script_name, script_name + '_epch_' + "{:03}".format(epoch + 1) + '.csv')
            save_df.to_csv(csv_save_path, index=False)
            print('Saved ' + csv_save_path)





