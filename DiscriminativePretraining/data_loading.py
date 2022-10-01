"""
(0) MinMaxScaler: Min Max normalizer
(1) real_data_loading: Load and preprocess real data
"""

## Necessary Packages
import numpy as np
import pandas as pd

def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


  
def real_data_loading (absolute_path):
    """Load and preprocess real-world datasets.
  
    Args:
    - absolute_path: absoulte_path
    - seq_len: sequence length

    Returns:
    - data: preprocessed data.
    """  
    ori_data = pd.read_csv(absolute_path, index_col= 0)
    cols = ["label", "subset"]
    ori_data = ori_data.drop(columns = cols)
    print(ori_data.head())
    ori_data = ori_data.values
    # First dimension is time or last dimension is label #ori_data = ori_data[:, :-1]
    ori_data = ori_data.astype(dtype = np.float64)
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)
    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(ori_data))    
    data = []
    for i in range(len(ori_data)):
        data.append(ori_data[idx[i]])

    return np.array(data)