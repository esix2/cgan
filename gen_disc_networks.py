import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim + 1, 512, 4, 1, 0, bias=False),  # Input is (noise_dim + 1, 1, 1), output is (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # Output is (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # Output is (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # Output is (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),  # Output is (1, 64, 64)
            nn.Tanh()
        )

    def forward(self, simulated_map, noise):
        noise = noise.view(-1, self.noise_dim, 1, 1)
        noise = noise.expand(-1, self.noise_dim, simulated_map.size(2), simulated_map.size(3))
        input = torch.cat((simulated_map, noise), 1)
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1, bias=False),  # Input is (2, 256, 256), output is (64, 128, 128)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # Output is (128, 64, 64)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # Output is (256, 32, 32)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # Output is (512, 16, 16)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # Output is (1, 13, 13) 
            nn.Conv2d(512, 1, 16, 1, 0, bias=False),  # Output is (1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

