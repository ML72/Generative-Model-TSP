import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(module_path)
from nets.level_generator import SeqVAE
from utils.distributions import link_batch



# Define constants
RESULT_FILENAME = 'vae_visualization.png'
NUM_SAMPLES = 4
NUM_NODES = 50

GEN_MODEL_PATH = 'pretrained/tsp_generator/vae_tsp_generator.pth'
HIDDEN_DIM = 512
LATENT_DIM = 128

# Seed both Numpy and PyTorch
np.random.seed(123)
torch.manual_seed(654)

# Load in VAE
generator = SeqVAE(input_dim=2, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, seq_length=NUM_NODES)
generator.load_state_dict(torch.load(GEN_MODEL_PATH, weights_only=True))
generator = generator.to('cuda' if torch.cuda.is_available() else 'cpu')
generator.eval()

# Utility function for plotting subplot
def subplot_embedding(subplot, graph, title, color):
    subplot.scatter(graph[:,0], graph[:,1], color=color)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title(title)

# Plot it!
fig, axs = plt.subplots(2, NUM_SAMPLES, figsize=(NUM_SAMPLES * 3, 2 * 3))
plt.subplots_adjust(hspace=0.1)
plt.tight_layout(rect=[0.01, 0, 0.97, 0.98])

LINK_VALUES = [5, 1, 10] # Out of order from [1, 5, 10, 15] for aesthetic reasons

vae_samples = generator.sample(num_samples=NUM_SAMPLES, seq_length=NUM_NODES).cpu().detach().numpy()
for i in range(NUM_SAMPLES):
    train_sample = link_batch(1, NUM_NODES, link_size=LINK_VALUES[i % len(LINK_VALUES)], noise=0.05)
    subplot_embedding(axs[0, i], train_sample[0], f"VAE Train Sample {i+1}", color='orange')
    subplot_embedding(axs[1, i], vae_samples[i], f"VAE Inference Sample {i+1}", color='blue')

if not os.path.exists("results/plots"):
    os.makedirs("results/plots")
plt.savefig(f"results/plots/{RESULT_FILENAME}", format='png')
plt.close()
