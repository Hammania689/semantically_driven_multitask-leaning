from collections import OrderedDict
from pathlib import Path
from PIL import ImageFile
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn

import datetime, time
import numpy as np
import pandas as pd
import re
import sys
import torch

# Ignore Pillow Warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Load Truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = ['airport_terminal',
 'apartment_building_outdoor',
 'arch',
 'auditorium',
 'barn',
 'beach',
 'bedroom',
 'castle',
 'classroom',
 'conference_room',
 'dam',
 'desert',
 'football_stadium',
 'great_pyramid',
 'hotel_room',
 'kitchen',
 'library',
 'mountain',
 'office',
 'phone_booth',
 'reception',
 'restaurant',
 'river',
 'school_house',
 'shower',
 'skyscraper',
 'supermarket',
 'waiting_room',
 'water_tower',
 'windmill']

def load_data(data_dir, batch_size=32, resize=64, train_ratio=.8, display=False):
    """Load dataset from specified path.
    Returns dataset and dataloader dictionaries with transforms set below.

    data_dir: String or Path object that specifies directory.
        Play it safe and use absolute path
    batch_size: Batch size of dataloader, 32 by default

    NOTE: Depending on your dataset folder names may have to change dictionary item names accordingly.
    i.e 'valid' to 'test' or even 'val'.
    """
    # Incorporate path checking and try catch in the future

    # Assumes order [Training, Testing]
    dataset_folders = {'train': '2D_Images_Training', 'test': '2D_Images_Testing'}

    if Path.exists(Path(data_dir)):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.RandomResizedCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {x[0]: datasets.ImageFolder(Path(data_dir, x[1]), data_transforms[x[0]]) for x in dataset_folders.items()}

        # Get size of train and split based on param train_ratio
        training_init_size = len(image_datasets['train'])
        train_size = int(train_ratio * training_init_size)
        val_size = training_init_size - train_size

        # Random Split and resulting grab indices
        train, val = random_split(image_datasets['train'], [train_size, val_size])
        train_indices = train.indices
        val_indices = val.indices

        if display:
            print(f'Initial Training Dataset size: {training_init_size}')
            print(f'Train/Val Ratio: {train_ratio}/{ 1 -train_ratio}')
            print(f'New Training Dataset size: {train_size}')

        # Create the dataloaders and store dataset_sizes
        train_dataloader = DataLoader(image_datasets['train'], batch_size=batch_size,
                                      sampler=SubsetRandomSampler(train_indices))
        val_dataloader = DataLoader(image_datasets['train'], batch_size=batch_size,
                                    sampler=SubsetRandomSampler(val_indices))
        test_dataloader = DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True, num_workers=4)

        dataloaders = {
            'train': train_dataloader,
            'val': val_dataloader,
            'test': test_dataloader
        }

        dataset_sizes = {
            'train': train_size,
            'val': val_size,
            'test': len(image_datasets['test'])
        }

        log = [x for x in dataloaders]
        print(f"Successfully Loaded built {log}")
        print(f"Root folders {dataloaders['train'].dataset.root} and {dataloaders['test'].dataset.root} ")
        return image_datasets, dataloaders, dataset_sizes
    else:
        print(f"{data_dir} is a faulty path")
        assert Path.is_dir(data_dir)


def save_model(model, optimizer, criterion=None, needed_epochs=None, model_params=None, scheduler=None, file_name=None):
    """
    Saves a model, optimizer state_dict and other passed parameters to a time stamped Checkpoint file.

    :param model:  Model to be saved
    :param optimizer: Optimizer used during training
    :param criterion: Loss function used during training (Default: None)
    :param needed_epochs: Number of needed to reach this result (Default: None)
    :param model_params: Model's current parameters (Default: None)
    :param scheduler:  Learning Rate scheduler (Default: None)
    :return:
    """
    # Send the model to cpu to save checkpoints
    model = model.to('cpu')

    # Grab the model's name and set (and if needed create) the checkpoint directory
    model_name = re.split("\(\n", str(model), 1)[0]
    parent_path = Path(Path.cwd(), model_name + "_checkpoints")
    if Path.exists(parent_path) == False:
        Path.mkdir(parent_path)

    # Time stamp the checkpoint file for later reference
    ts = datetime.datetime.fromtimestamp(time.time()).strftime('%b%d|%H:%M:%S')
    if file_name is None:
        file_name = "checkpoint" + ts + ".pth"

    # Add model info to checkpoint dictionary
    checkpoint = {'optimizer': optimizer.state_dict(),
                  'model_params': model_params,
                  'criterion': criterion,
                  'epochs': needed_epochs,
                  'state_dict': model.state_dict()}

    if isinstance(model, nn.DataParallel):
        checkpoint['state_dict'] = model.module.state_dict()

    if model_params is not None:
        checkpoint['model_params'] = model_params
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()

    # Save the file
    torch.save(checkpoint, Path(parent_path, file_name))


def log_params(data=None, file_path='ExperimentLog.csv', columns=None, OVERRIDE=False):
    """
    Use to save hyper parameters in Excel for future reference

    data: Hyper parameter Information for future reference and comparison
    file_path: desired path to save Excel file
        By default saves to 'ExperimentLog.csv' in current directory

    columns: reference labels for records in the Excel file. Default = None
    """
    if not OVERRIDE:
        if data.shape != (1, len(columns)):
            if data is None:
                data = np.arange(len(columns))
            data = np.reshape(data, (1, len(columns)))

    # Create dataframe if spreadsheet doesn't exist
    if Path.exists(Path(file_path)) == False:
        param_log = pd.DataFrame(data, columns=columns)
        print(f"{param_log}")
        pd.DataFrame.to_csv(param_log, file_path, columns=columns, index=False)
    else:
        # Otherwise load the existing data and append to the end
        param_log = pd.DataFrame(data, columns=columns)
        pd.DataFrame.to_csv(param_log, file_path, header=False, index=False, mode='a')

    if Path.exists(Path(file_path)):
        print(f"Hyper Parameters have been written to {file_path}")


def load_module_checkpoint(checkpoint):
    """
    Use to fix and load  Data parrallel protected state_dict saves
    :return: check_points: dictionary of the corrected model save state, parameters, etc
    :param checkpoint: path to the checkpoint file that needs to be loaded
    """
    corrected_state_dict = OrderedDict()

    # Remove module from each key in the dictionary
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]
        if 'module' in k[:7]:
            corrected_state_dict[name] = v

    checkpoint['state_dict'] = corrected_state_dict
    return checkpoint


def load_model(model, checkpoint_path, Display=False):
    """ Restores the model to the saved checkpoint path.
        If model is nn.DataParallel calls appropriate function to handle

        model: Model to restored
        best_model_wts: The model's state_dict()

        Returns model with loaded weights.
    """
    module_in_dict = False

    if Path.exists(Path(checkpoint_path)) == False:
        print("Faulty path to checkpoint")
        return None

    checkpoint = torch.load(checkpoint_path, device)

    # Check for module in dictionary
    for key, val in checkpoint.items():
        if 'module' in key[:7]:
            module_in_dict = True
        if Display:
            print(f'{key}:{val}')

    if module_in_dict:
        checkpoint = load_module_checkpoint(checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])
    return model