#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from gen_disc_networks import Generator, Discriminator
from customDataSet import CustomDataset, get_data_loaders
from evaluate_rmse import evaluate_rmse
from IPython.display import Image


# # Creating a cGAN model

# ## Definiting the parameters, Generaor and Discriminator networks

# In[2]:

def tst():
    print("This is a test.")
def main(): 
    import torch

# Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = "cpu"
# Hyperparameters

    pixel_size = 256
    batch_size = 64
    noise_dim = 2
    lr = 0.002
    num_epochs = 20
    ifTrain = True
    ifSaveModel = False
    ifCalcRMSE = False


    G = Generator(noise_dim)
    D = Discriminator()
#G.get_submodule

    generator = G.to(device)
    discriminator = D.to(device)


# ## Getting the model summary

# In[3]:


##print("Generator summary")
#summary(G, [(1, pixel_size, pixel_size), (noise_dim, pixel_size, pixel_size)])
#
#print("Discriminator summary")
#summary(D, (2, 256, 256))
##for parameter in G.parameters():
##    print(parameter)
## Initialize models
#


# ## Splitting the dataset to train and test set

# In[4]:


# Create dataset instance

    HOME = os.environ["HOME"]
    dataset_dpm_path = f"{HOME}/data/raytracing-dataset/DPM/"
    dataset_irt_path = f"{HOME}/data/raytracing-dataset/IRT2/"


# Create data loader
#data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_loader, test_loader = get_data_loaders(dataset_dpm_path, dataset_irt_path, batch_size, test_split=0.99)

# Choose a random index
    random_index = random.randint(0, len(test_loader.dataset) - 1)
# Get the image at the random index

    image_dpm, image_irt = test_loader.dataset[random_index]

# Convert tensor to numpy array and remove batch dimension
    image_dpm_np = image_dpm.squeeze().numpy()
    image_irt_np = image_irt.squeeze().numpy()

#plt.imshow(image_dpm_np, cmap='gray')
#plt.axis('off')
#plt.show()
#print(len(test_loader.dataset))
    print(f"The size of training set: {len(train_loader.dataset)}")


# In[ ]:





# ## Training the model

# In[5]:


    if ifTrain == True:
        
        # Define loss function and optimizers
        criterion = nn.BCELoss()
        d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Training loop
        for epoch in range(num_epochs):
            for i, batch in enumerate(tqdm(train_loader)):
                # Unpack the batch
                simulated_map, measured_map = batch
#            print(simulated_map.size())
        
                # Move tensors to device
                simulated_map = simulated_map.to(device)
                measured_map = measured_map.to(device)
        
        
                ############################
                # Train discriminator
                ############################
                d_optimizer.zero_grad()
        
        
        
                # Train with real data
                real_labels = torch.ones(batch_size, 1, device=device)
                # Concatenate the simulated map and measured map along the channel dimension
                real_inputs = torch.cat((simulated_map, measured_map), dim=1)
                real_outputs = discriminator(real_inputs)
        #        print(f"real label: {real_labels.size()}")
        #        print(f"real output: {real_outputs.size()}")
                d_loss_real = criterion(real_outputs.squeeze(), real_labels.squeeze())
                d_loss_real.backward()
        
                # Train with fake data
                noise_map = torch.randn(batch_size, noise_dim, pixel_size, pixel_size, device=device)
                fake_images = generator(simulated_map, noise_map)
                fake_labels = torch.zeros(batch_size, 1, device=device)
        #        print(f"fake image size @generator's output: {fake_images.size()}")
        #        print(f"simulated_map size: {simulated_map.size()}")
                fake_inputs = torch.cat((simulated_map, fake_images), dim=1)
                fake_outputs = discriminator(fake_inputs)
        #        print(f"fake label: {fake_labels.size()}")
        #        print(f"fake output: {fake_outputs.size()}")
                d_loss_fake = criterion(fake_outputs.squeeze(), fake_labels.squeeze())
                d_loss_fake.backward()
        
                d_loss = d_loss_real + d_loss_fake
                d_optimizer.step()
        
                ############################
                # Train generator
                ############################
                g_optimizer.zero_grad()
        
                # Generate fake images
                noise_map = torch.randn(batch_size, noise_dim, pixel_size, pixel_size, device=device)
                fake_images = generator(simulated_map, noise_map)
        
                # Train generator with discriminator feedback
                fake_inputs = torch.cat((simulated_map, fake_images), dim=1)
                outputs = discriminator(fake_inputs)
                g_loss = criterion(outputs.squeeze(), real_labels.squeeze())
                g_loss.backward()
                g_optimizer.step()
        
        
        
                ############################
                # Print losses
                ############################
                if i % 1 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                          f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
                    if ifSaveModel == True:
                        torch.save(discriminator.state_dict(), 'discriminator_trained.pth')
                        torch.save(discriminator, 'discriminator_entire_model.pth')
                        torch.save(generator.state_dict(), 'generator_trained.pth')
                        torch.save(generator, 'generator_entire_model.pth')
                        print(f"Model saved at epoch {i}")


# # Evaluating the results

# ## Showing a sample synthetic map

# In[ ]:


    generator = Generator(noise_dim).to(device)
    generator.load_state_dict(torch.load('generator_trained.pth'))
#generator.eval()


# In[ ]:


# Generate a new image
#from synthetic_map import generate_image
    def generate_image(generator, simulated_map, noise_dim):
        # Prepare simulated map and noise map
        simulated_map = simulated_map.to(device)
        noise_map = torch.randn(1, noise_dim, pixel_size, pixel_size, device=device)

        # Generate image
        with torch.no_grad():
            generated_image = generator(simulated_map, noise_map)
       # Denormalize the generated image from [-1, 1] to [0, 1]
#    generated_image = (generated_image + 1) / 2 
        return generated_image

# Choose a random index
    random_index = random.randint(0, len(test_loader.dataset) - 1)
# Get the image at the random index

    image_dpm, image_irt = test_loader.dataset[random_index]

# Convert tensor to numpy array and remove batch dimension
    image_dpm_np = image_dpm.squeeze().numpy()
    image_irt_np = image_irt.squeeze().numpy()
    simulated_map = torch.from_numpy(image_dpm_np).reshape(1, 1 , pixel_size, pixel_size)
    synthetic_image = generate_image(generator, simulated_map, noise_dim).numpy()
    synthetic_image = synthetic_image.squeeze()
    print(synthetic_image.shape)
#print(generated_image.size())

# Plot the image

    plt.imshow(image_irt_np , cmap='gray')
    plt.axis('off')
    plt.show()

    plt.imshow(image_dpm_np , cmap='gray')
    plt.axis('off')
    plt.show()

    plt.imshow(synthetic_image, cmap='gray')
    plt.axis('off')
    plt.show()


# ## Calculate the RMSE on the test set

# In[ ]:


    if ifCalcRMSE:
        avg_rmse_generated, avg_rmse_simulated = evaluate_rmse(generator, test_loader, noise_dim)
        print(f"Average RMSE for generated images: {avg_rmse_generated}")
        print(f"Average RMSE for simulated maps: {avg_rmse_simulated}")


# ## Visualizing the generator

# In[ ]:


    import torch
    from torchviz import make_dot
    import torch.onnx


# Create a dummy input for the generator (batch size 1)
    dummy_input = torch.randn(1, 1, 256, 256)
# Generate the visualization
    noise_map = torch.randn(1, noise_dim, pixel_size, pixel_size, device=device)
    output = generator(dummy_input, noise_map)
    model_viz = make_dot(output, params=dict(generator.named_parameters()))
# Render the graph to a file
    model_viz.render("generator_architecture", format="png")

# Display the graph inline (optional, for Jupyter/Colab)
    from IPython.display import Image
    Image(filename="generator_architecture.png")



# ## Visualizing the discriminator

# In[ ]:


# Create a dummy input for the discriminator (batch size 1, single-channel grayscale image)
    dummy_input = torch.randn(1, 2, 256, 256)

# Generate the visualization
#real_inputs = torch.cat((simulated_map, measured_map), dim=1)
    output = discriminator(dummy_input)
    model_viz = make_dot(output, params=dict(discriminator.named_parameters()))

# Render the graph to a file
    model_viz.render("discriminator_architecture", format="png")

# Display the graph inline (optional, for Jupyter/Colab)
    Image(filename="discriminator_architecture.png")


if __name__ == "__main__":
    main()
