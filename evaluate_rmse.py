import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm

def calculate_rmse(img1, img2):
    return torch.sqrt(F.mse_loss(img1, img2))

def evaluate_rmse(generator, test_loader, noise_dim):
    pixel_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.eval()
    
    total_rmse_generated = 0.0
    total_rmse_simulated = 0.0
    num_samples = 0

    with torch.no_grad():
        j = 0
        for simulated_map, measured_map in tqdm(test_loader):
            j += 1
            print(f"j: {j}")
            simulated_map = simulated_map.to(device)
            measured_map = measured_map.to(device)
            batch_size = simulated_map.size(0)

        
            # Generate synthetic images
#            with torch.no_grad():
            noise_map = torch.randn(batch_size, noise_dim, pixel_size, pixel_size, device=device)
            generated_images = generator(simulated_map, noise_map)

            # Denormalize the generated images from [-1, 1] to [0, 1]
            generated_images = (generated_images + 1) / 2

            # Denormalize the simulated and measured maps from [-1, 1] to [0, 1]
            simulated_map_denorm = (simulated_map + 1) / 2
            measured_map_denorm = (measured_map + 1) / 2

            # Calculate RMSE for the batch
            for i in range(batch_size):
                rmse_generated = calculate_rmse(generated_images[i], measured_map_denorm[i])
                rmse_simulated = calculate_rmse(simulated_map_denorm[i], measured_map_denorm[i])
                
                total_rmse_generated += rmse_generated.item()
                total_rmse_simulated += rmse_simulated.item()
                num_samples += 1
            if j > 5: break

    avg_rmse_generated = total_rmse_generated / num_samples
    avg_rmse_simulated = total_rmse_simulated / num_samples

    return avg_rmse_generated, avg_rmse_simulated
