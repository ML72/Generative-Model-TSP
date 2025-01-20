import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(module_path)
from nets.level_generator import SeqVAE
from utils.distributions import link_batch



# Define constants
RESULT_FILENAME = 'vae_coverage.png'
NUM_SAMPLES = 150
NUM_NODES = 50

GEN_MODEL_PATH = 'pretrained/tsp_generator/vae_tsp_generator.pth'
HIDDEN_DIM = 512
LATENT_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Seed both Numpy and PyTorch
np.random.seed(123)
torch.manual_seed(654)

# Load in VAE
generator = SeqVAE(input_dim=2, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, seq_length=NUM_NODES)
generator.load_state_dict(torch.load(GEN_MODEL_PATH, weights_only=True))
generator = generator.to(DEVICE)
generator.eval()

# Sample data
LINK_VALUES = [1, 5, 10, 15]
vae_samples = generator.sample(num_samples=NUM_SAMPLES, seq_length=NUM_NODES)
train_samples = np.zeros((NUM_SAMPLES, NUM_NODES, 2))
for i in range(NUM_SAMPLES):
    train_samples[i] = link_batch(1, NUM_NODES, link_size=LINK_VALUES[i % len(LINK_VALUES)], noise=0.05)[0]

# Encode data and conduct PCA
vae_samples = vae_samples.float().to(DEVICE)
train_samples = torch.tensor(train_samples).float().to(DEVICE)
vae_samples_z = generator.encode(vae_samples).cpu().detach().numpy()
train_samples_z = generator.encode(train_samples).cpu().detach().numpy()

pca = PCA(n_components=2)
pca.fit(vae_samples_z)
graph = np.concatenate((vae_samples_z, train_samples_z), axis=0)
graph = pca.transform(graph)

vae_embed = graph[:NUM_SAMPLES]
train_embed = graph[NUM_SAMPLES:]

# Plot it!
plt.scatter(vae_embed[:,0], vae_embed[:,1], color='blue', label='VAE Inference Samples')
plt.scatter(train_embed[:,0], train_embed[:,1], color='orange', label='VAE Training Samples')
plt.xticks([])
plt.yticks([])
plt.title('Latent Space Coverage of VAE Samples')
plt.legend()

if not os.path.exists("results/plots"):
    os.makedirs("results/plots")
plt.savefig(f"results/plots/{RESULT_FILENAME}", format='png')
plt.close()
