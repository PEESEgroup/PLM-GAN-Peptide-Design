import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from esm.modules import ESM1bLayerNorm, TransformerLayer


class CoordinateChannel1D(nn.Module):
	def __init__(self):
		super(CoordinateChannel1D, self).__init__()
	
	def forward(self, x):
		#  (batch_size, channels, seq_len)
		batch_size, _, seq_len = x.size()
		coords = torch.linspace(-1, 1, steps=seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1).unsqueeze(1)
		return torch.cat([x, coords], dim=1)  


class generatorNet(nn.Module):
	def __init__(self, latent_dim = 100, output_shape = (12, 18), conv_depth = 6, conv_channels = 128):
		super(generatorNet, self).__init__()
		self.output_shape = output_shape
		
		# FC
		self.fc = nn.Linear(latent_dim, np.prod(output_shape))
		self.leaky_relu = nn.LeakyReLU(0.2)
		
		# CoordinateChannel1D
		self.coord_channel = CoordinateChannel1D()
		
		self.conv_depth = nn.ModuleList([
			nn.Sequential(
				nn.Conv1d(in_channels=output_shape[1] + 1 if i == 0 else conv_channels, 
						  out_channels=conv_channels, 
						  kernel_size=3, 
						  dilation=2 ** i, 
						  padding=2 ** i),
				nn.LeakyReLU(0.2)
			) for i in range(conv_depth)
		])
		
		self.final_conv = nn.Conv1d(in_channels=conv_channels * conv_depth, 
									out_channels=output_shape[1],  # channels
									kernel_size=1, 
									padding=0)
		
	def forward(self, x):
		x = self.fc(x)
		x = self.leaky_relu(x)
		x = x.view(x.size(0), self.output_shape[0], self.output_shape[1])
		x = x.permute(0, 2, 1)  # out: [batch_size, num_channels, sequence_len]
		x = self.coord_channel(x)
		
		total_features = []
		for block in self.conv_depth:
			x = block(x)
			total_features.append(x)
		
		x = torch.cat(total_features, dim=1)
		
		x = self.final_conv(x)
		x = x.permute(0, 2, 1)  # out: [batch_size, sequence_len, num_channels]
		
		return x


class criticNet(nn.Module):
	def __init__(self, conv_depth = 6, filters = 64,  dense_width = 256):
		super(criticNet, self).__init__()

		self.coord_channel = CoordinateChannel1D()
		self.spatial_dropout = nn.Dropout2d(p=0.25) 
		
		self.conv_layers = nn.ModuleList([
			nn.Sequential(
				nn.Conv1d(in_channels=filters if i > 0 else 18 + 1, 
						  out_channels=filters,
						  kernel_size=4,
						  stride=2,
						  padding=2),
				nn.LeakyReLU(0.2)
			)
			for i in range(conv_depth)
		])
		
		self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
		
		self.dense_layers = nn.ModuleList([
			nn.Sequential(
				nn.Dropout(p=0.25),
				nn.Linear(filters if i == 0 else dense_width, dense_width),
				nn.LeakyReLU(0.2)
			)
			for i in range(3)
		])
		
		self.output_layer = nn.Linear(dense_width, 1)
		
	def forward(self, x):

		x = x.permute(0, 2, 1)  # out: [batch_size, num_channels, sequence_len] 
		x = self.coord_channel(x)
		
		for conv in self.conv_layers:
			x = self.spatial_dropout(x.unsqueeze(2)).squeeze(2)
			x = conv(x)
		
		x = self.global_avg_pool(x).squeeze(-1)
		
		for dense in self.dense_layers:
			x = dense(x)
		
		x = self.output_layer(x)
				
		return x



class predictorNet(nn.Module):
	def __init__(self):
		super(predictorNet, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
		self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
		self.fc1 = nn.Linear(32 * 80, 256) 
		self.fc2 = nn.Linear(256, 64)
		self.fc3 = nn.Linear(64, 1) 

	def forward(self, x):
		x = x.unsqueeze(1)
		x = F.relu(self.conv1(x))
		x = self.pool(x)
		x = F.relu(self.conv2(x))
		x = self.pool(x)
		
		x = x.view(x.size(0), -1)
		
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)  
		
		return x



class modifyESM2(nn.Module):
	def __init__(
		self,
		num_layers: int = 8, 
		embed_dim: int = 320, 
		attention_heads: int = 20,
		alphabet_size: int = 20
	):
		super().__init__()
		self.num_layers = num_layers
		self.embed_dim = embed_dim
		self.attention_heads = attention_heads
		self.alphabet_size = alphabet_size

		self.embed_tokens = nn.Embedding(
			self.alphabet_size,
			self.embed_dim,
		)

		self.layers = nn.ModuleList(
			[
				TransformerLayer(
					self.embed_dim,
					4 * self.embed_dim,
					self.attention_heads,
					add_bias_kv=False,
					use_esm1b_layer_norm=True,
					use_rotary_embeddings=True,
				)
				for _ in range(self.num_layers)
			]
		)

		self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)


	def forward(self, onehot_tokens):
		x = torch.matmul(onehot_tokens, self.embed_tokens.weight)

		cls_weight = self.embed_tokens.weight[0].unsqueeze(0).unsqueeze(0).repeat(x.size(0), 1, 1)
		eos_weight = self.embed_tokens.weight[1].unsqueeze(0).unsqueeze(0).repeat(x.size(0), 1, 1)
		x = torch.cat((cls_weight, x, eos_weight), dim=1)

		# (B, T, E) => (T, B, E)
		x = x.transpose(0, 1)

		for layer in self.layers:
			x, attn = layer(
				x,
				self_attn_padding_mask=None,
				need_head_weights=False,
			)

		x = self.emb_layer_norm_after(x)
		x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

		return torch.mean(x[:, 1:-1, :], dim=1) # delete cls and eos embedding


def gradient_penalty(critic, real_samples, fake_samples):

    batch_size = real_samples.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=real_samples.device)
    interpolates = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolates = interpolates.requires_grad_(True)
    critic_interpolates = critic(interpolates)
    
    gradients = grad(outputs=critic_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones_like(critic_interpolates), 
                     create_graph=True, retain_graph=True)[0]
    
    gradients = gradients.contiguous().view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


def wasserstein_loss(fake_output, real_output=None):
    if real_output is not None:
        # Critic loss: Maximize real_output - fake_output
        return torch.mean(fake_output) - torch.mean(real_output)
    else:
        # Generator loss: Minimize fake_output
        return -torch.mean(fake_output)