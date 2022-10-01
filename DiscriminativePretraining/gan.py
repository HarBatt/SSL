import torch 
import torch.nn as nn
from utils import batch_generator


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x 

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        return x


def gan(ori_data, parameters):
    generator = Generator(parameters.latent_dim, parameters.hidden_dim, parameters.feat).to(parameters.device)
    discriminator = Discriminator(parameters.feat, parameters.hidden_dim, 1).to(parameters.device)

    
    # Optimizers for the models, Adam optimizer with learning rate = 0.001 for the generator and SGD with learning rate of 0.001 for the discriminator.
    gen_optim = torch.optim.Adam(generator.parameters(), lr=0.001)
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr = 0.001)

    # Batch generator, it keeps on generating batches of data.
    data_gen = batch_generator(ori_data, parameters)
    # BCE with logits
    criterion = torch.nn.BCEWithLogitsLoss()

    for step in range(parameters.iterations):
        for disc_step in range(parameters.disc_extra_steps):
            """
            Discriminator training.
            
            - Generate fake data from the generator.
            - Train the discriminator on the real data and the fake data.

            Note: Make sure to detach the variable from the graph to prevent backpropagation. 
                in this case, it is the synthetic data, (generator(noise)).
            """
            # Get the real batch data, and synthetic batch data. 
            bdata = data_gen.__next__() 
            noise = torch.randn(parameters.batch_size, parameters.latent_dim).to(parameters.device)

            fake = generator(noise).detach() 

            fake_dscore = discriminator(fake)
            true_dscore = discriminator(bdata)

            # Compute the loss for the discriminator, and backpropagate the gradients.
            dloss = criterion(fake_dscore, torch.zeros_like(fake_dscore)) + criterion(true_dscore, torch.ones_like(true_dscore))
            disc_optim.zero_grad()
            dloss.backward()
            disc_optim.step()

        noise = torch.randn(parameters.batch_size,  parameters.latent_dim).to(parameters.device)
        fake = generator(noise) 
        fake_dscore = discriminator(fake)

        # Compute the loss for the generator, and backpropagate the gradients.
        gloss = criterion(fake_dscore, torch.ones_like(fake_dscore))

        gen_optim.zero_grad()
        gloss.backward()
        gen_optim.step()

        if step % parameters.print_steps == 0:
            print('[Step {}; L(G): {}; L(D): {}]'.format(step, dloss, gloss))


    torch.save(generator, parameters.pre_train_path + 'generator.mdl')
    torch.save(discriminator, parameters.pre_train_path + 'discriminator.mdl')

    noise = torch.randn(ori_data.shape[0], parameters.latent_dim).to(parameters.device)
    synthetic_samples = generator(noise).detach().cpu().numpy()
    return synthetic_samples



