from pathlib import Path
from models import *
from ops.lstd import LSTD
from tensorboardX import SummaryWriter
from torch import nn
from utils.wordnet_similarity import get_scene_synset_dictionary, exhuastive_lesk_simarity_metric
from utils.data import load_data, log_params, save_model, classes

import re
import argparse
import copy
import datetime
import time
import pandas as pd

parser = argparse.ArgumentParser(description='Test Semantic VAE Classifer')
parser.add_argument('--batch_size', type=int, default=32, help='Number of images per batch')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--chp_path',  default='checkpoint.pth', help='path to the model checkpoint that will be loaded to test')

# Classes within the dataset
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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print(args)

vae = VAE()
batch_size = args.batch_size
vae_checkpoints = torch.load("BN_best.pth", device)
vae.load_state_dict(vae_checkpoints['state_dict'])
vae = vae.to(device)

# Load Dataset, Dataloaders, etc
shrec_sub_path = 'Data/Scene_SBR_IBR_Dataset/'
data_dir = Path(Path.home(), shrec_sub_path)
dataset, dataloader, dataset_sizes = load_data(data_dir, batch_size=batch_size)


# Set Summary writer for Tensorboard
ts = datetime.datetime.fromtimestamp(time.time()).strftime('%b%d|%H:%M:%S')
tb_logdir = 'runs/test_' + ts
writer = SummaryWriter(log_dir=tb_logdir)

# Initialized variable for logging information
since = time.time()
corrects = 0
total = 0
training_epoch_time = 0
pred_label_name = np.array([])
pred_labels = np.array([])
pred_prob = np.array([])
correct_label = np.array([])
correct_label_name = np.array([])


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = ((std * image)) + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)

    return ax

vae.eval()
with torch.no_grad():
    for data in dataloader['test']:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward Pass
        recon_images, mu, logvar, class_pred = vae.forward(inputs, labels)

        # Class prediction and number of instance
        _, pred = torch.max(class_pred.data, 1)
        top_prob, top_pred = torch.topk(class_pred, 1, dim=1)

        top_prob, top_pred = top_prob.cpu().numpy(), top_pred.cpu().numpy()
        correct_label = np.append(correct_label, labels.cpu().numpy())
        correct_label_name = np.append(correct_label_name,
                                       np.array([classes[x] for x in np.nditer(labels.cpu().numpy())]))

        # if idx % 1000 == 0:
        #     writer.add_images("Images", inputs)
        #     writer.add_images("Reconstructions", recon_images)
        #     writer.add_graph(VAE().cuda(), (inputs, labels))

        # Save result for reference
        if pred_labels.size == 0:
            # Append
            pred_labels = np.append(pred_labels, top_pred)
            class_names = np.array([classes[x] for x in np.nditer(top_pred)])
            pred_label_name = np.append(pred_label_name, class_names)
            pred_prob = np.append(pred_prob, top_prob)

            # Reshape
            pred_labels = np.reshape(pred_labels, top_pred.shape)
            pred_label_name = np.reshape(pred_label_name, top_pred.shape)
            pred_prob = np.reshape(pred_prob, top_prob.shape)
        else:
            # Stack the arrays accordingly
            pred_labels = np.row_stack((pred_labels, top_pred))

            class_names = np.array([classes[x] for x in np.nditer(top_pred)])
            class_names = np.reshape(class_names, top_pred.shape)
            pred_label_name = np.row_stack((pred_label_name, class_names))

            pred_prob = np.row_stack((pred_prob, top_prob))


        # (Classifer) Average Loss and Accuracy for current batch
        total += labels.size(0)
        corrects += torch.sum(pred == labels.data)

# Calculate Accuracy and time since model began testing
classifier_acc = corrects.double() / total
time_elapsed = time.time() - since


# Print out results, accuracy as well as any other information
print(f'Accuracy of the network on the {dataset_sizes} test images: {classifier_acc * 100}%')
print(f'Testing complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
writer.close()


# vae.eval()
# with torch.no_grad():
#     images, label = next(iter(dataloader['test']))
#
#     images = images.to(device)
#     label = label.to(device)
#
#     recon_images, mu, logvar, class_pred = vae.forward(images, label)
#     _, pred = torch.max(class_pred.data, 1)
#     top_prob, top_pred = torch.topk(class_pred, 1, dim=1)
#
#     comparison = torch.cat([images.cpu(), recon_images.cpu()])
#     file_name = 'reconstruction.png'
#     print(comparison[0].view(3, 64, 64))
# print(file_name)
# save_image(comparison[0].cpu(), './test.png')
#
#
#

