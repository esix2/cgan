import torch
from torchvision.utils import save_image

# Function to generate a new image
def generate_image(generator, simulated_map, noise_dim):
    # Prepare simulated map and noise map
    simulated_map = simulated_map.to(device)
    noise_map = torch.randn(1, noise_dim, 256, 256).to(device)

    # Generate image
    with torch.no_grad():
        generated_image = generator(simulated_map, noise_map).squeeze(0)
    
    return generated_image

