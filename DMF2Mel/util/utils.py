import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import math

def get_writer(output_directory, log_directory):

    logging_path=f'{output_directory}/{log_directory}'
    if os.path.exists(logging_path) == False:
        os.makedirs(logging_path)
    writer = CustomWriter(logging_path)
            
    return writer

class CustomWriter(SummaryWriter):
    def __init__(self, log_dir):
        super(CustomWriter, self).__init__(log_dir)
        
    def add_losses(self, name, phase, loss, global_step):
        self.add_scalar(f'{name}/{phase}', loss, global_step)
        

def save_checkpoint(model, optimizer, learning_rate, epoch, filepath):
    print(f"Saving model and optimizer state at iteration {epoch} to {filepath}")
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, f'{filepath}/checkpoint_{epoch}')
    
