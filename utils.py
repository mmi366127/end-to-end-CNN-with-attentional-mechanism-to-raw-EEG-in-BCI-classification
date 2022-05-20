from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import matplotlib.pyplot as plt

import torch, scipy.io
import numpy as np

# Object for saving training history
class history():
    def __init__(self):
        
        self.parameters = {
            'key':'value'
        }

    
# Parse the parameters of the parser
def parser_parameters(parser):

    pass


# plot for loss or accuracy curve
def plot_curve(log_array, title = 'loss'):
    
    x = np.arange(log_array.shape[1])
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('train ' + title, color = color)

    ax1.plot(x, log_array[0], color = color)
    ax1.tick_params(axis = 'y', labelcolor = color)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('valid ' + title, color = color)
    ax2.plot(x, log_array[1], color = color)
    ax2.tick_params(axis = 'y', labelcolor = color)

    fig.tight_layout()
    plt.show()










