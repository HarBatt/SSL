## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# 1. RGAN model
from gan import gan
# 2. Data loading
from data_loading import real_data_loading
# 3. Utils
from utils import Parameters
#from metrics.visualization_metrics import visualization

#set the device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Data loading
data_path = "data/"
path_real_data = data_path + 'clean_data.csv'
#Evaluation of the model, by default can be set to false.
eval_model = False
#parameters
parameters = Parameters()

"""
Parameters for the GAN model.
---------------------------------------------------------------------------------------------------------------------
1) batch_size: Batch size for training.
2) hidden_dim: Number of hidden units in the recurrent cell.
3) num_layer: Number of layers in the recurrent cell.
4) iterations: Number of epochs to train the model.
"""

parameters.pre_train_path = "pre_trained_model/" # Path to save the pre-trained model.
parameters.batch_size = 64   # Batch size for training.
parameters.hidden_dim = 96 #Hidden dim for the LSTM
parameters.iterations = 20000 # Number of epochs to train the model.
parameters.latent_dim = 32 #Latent dim for the generator
parameters.disc_extra_steps = 1 #Extra steps for the discriminator
parameters.feat = 196  #Number of features
parameters.device = device #Device
parameters.print_steps = 500 #Device

#preprocessing the data.

"""
Method: real_data_loading()
---------------------------------------------------------------------------------------------------------------------
    - Loads the data from the path.
    - Scales the data using a min-max scaler.
    - Slices the data into windows of size seq_len.
    - Shuffles the data randomly, and returns it.

"""
ori_data = real_data_loading(path_real_data)   
print('Preprocessing Complete!')

with open(data_path + 'real_data.npy', 'wb') as f:
    np.save(f, np.array(ori_data))

print("Saved real data!")

# Run GAN
"""  
Method: gan()
---------------------------------------------------------------------------------------------------------------------
    - Runs the gan model.
"""
generated_data = gan(ori_data, parameters)   
print('GAN Training Complete!')

with open(data_path + 'synthetic_data.npy', 'wb') as f:
    np.save(f, np.array(generated_data))

