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

parser = argparse.ArgumentParser(description='Train Semantic VAE Classifer')
parser.add_argument('--batch_size', type=int, default=32, help='Number of images per batch')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--Lambda', type=float, default=.1, help='Weight that will applied to the LSTD Loss')
parser.add_argument('--arch', default='vae', help="Model Arch to be used.")
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--metric', default="lesk", help='WordNet Similarity metric to use')
parser.add_argument('--display', default=False, help='Shows progress between epochs')
parser.add_argument('--data_path', default='/home/hameed/Data/Scene_SBR_IBR_Dataset/', help='Path to the data directory')
parser.add_argument('--lesk_path', default='lesk_scores.csv', help='CSV with stored Lesk scores')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
device = torch.device("cuda:0" if args.cuda else "cpu")

# Define Model and Hyperparameters
archs = {'vae': VAE(), 'vae_bn': VAE_BN(), 'vae_gn': VAE_GN()}
model = archs[args.arch]
model.name = re.split("\(\n", str(model), 1)[0]
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=7, gamma=0.1)
batch_size = args.batch_size
epochs = args.epochs
Lambda = args.Lambda

# Initialize logging variables
since = time.time()
best_model_wts = 0
best_acc = 0.0
needed_epochs = 0
training_epoch_time = 0
model_params = 0
col_headers = ['time', 'accuracy', 'total_loss', 'classification_loss',
               'reconstruction_loss', 'lstd_loss', 'semantic similarity metric' , 'Lambda', 'classifier_architecture',
               'epochs', 'batch_size', 'optimizer', 'init_lr', 'scheduler']
ts = datetime.datetime.fromtimestamp(time.time()).strftime('%b%d|%H:%M:%S')
log_file = args.arch + "_ExperimentLog.csv"
best_run_path = "Best|" + ts + ".pth"

# Set Summary writer for Tensorboard
tb_logdir = 'runs/' + ts
writer = SummaryWriter(log_dir=tb_logdir)

# Information to be logged
optim_name = re.split(' ', str(optim), maxsplit=1)[0]
scheduler_name = str(scheduler).split('.')[3].split(' ')[0]
init_lr = optim.param_groups[0]['lr']

# Load Dataset, Dataloaders, etc
# Grab the stored Lesk scores
# Get and Store WordNet Synsets of each class
print(args)
print("Training ", model.name)
dataset, dataloader, dataset_sizes = load_data(Path(args.data_path), batch_size=batch_size)
lesk_scores = pd.read_csv(args.lesk_path) if Path(args.lesk_path) else exhuastive_lesk_simarity_metric(classes)
scene_synsets = get_scene_synset_dictionary(classes)

# Each epoch has a training and validation phase
phases = ['train', 'val']
for epoch in range(epochs):
    for phase in phases:
        if phase == 'train':
            scheduler.step()
            model.train()  # Set model to training mode
            model = model.to(device)
        else:
            model.eval()  # Set model to evaluate mode
            model = model.to(device)

        running_cls_loss = 0.0
        running_corrects = 0
        running_recon_loss = 0.0
        running_bce_loss = 0.0
        running_kld_loss = 0.0
        running_lstd_loss = 0.0
        running_total_loss = 0.0

        epoch_since = time.time()

        for idx, (inputs, labels) in enumerate(dataloader[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Reset parameter gradients for Optimizer
            optim.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                recon_images, mu, logvar, class_pred = model.forward(inputs, labels)

                # Get class labels/predictions and adjust gradient
                _, predicted = torch.max(class_pred, 1)

                # Calculate each normalized loss
                reconstruction_loss, bce, kld = SE_loss(recon_images, inputs, mu, logvar)
                lstd_loss = LSTD(model.scene_dist, model.eps, labels, lesk_scores=lesk_scores, device=device, metric=args.metric, scene_synsets=scene_synsets, display=args.display)
                classifier_loss = criterion(class_pred, labels)

                # Weighted sum of each loss into a total loss function
                reconstruction_loss = (1 - Lambda) * reconstruction_loss
                lstd_loss = Lambda * lstd_loss
                classifier_loss = (1 - Lambda) * classifier_loss

                total_loss = reconstruction_loss + lstd_loss + classifier_loss

                # Backprop Total loss (VAE + Classifer) and update VAE if training
                if phase == 'train':
                    # Calculate Total loss and Backprop through network
                    total_loss.backward()
                    optim.step()

                if idx % 1000 == 0:
                    writer.add_images("Images", inputs, global_step=epoch)
                    writer.add_images("Reconstructions", recon_images, global_step=epoch)
                    # writer.add_graph(VAE(), (inputs, labels), global_step=epoch)

            if idx % 100 == 0 and args.display:
                print(f'{phase} Batch Loss: {running_cls_loss}| Acc: {running_corrects / batch_size}')

            # (Classifer) Average Loss and Accuracy for current batch
            running_cls_loss += classifier_loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted == labels.data)

            # (VAE) Average Losses for current batch
            running_recon_loss += reconstruction_loss.item() * inputs.size(0)
            running_bce_loss += bce.item() * inputs.size(0)
            running_kld_loss += kld.item() * inputs.size(0)

            # (LSTD) Average Losses for current batch
            running_lstd_loss += lstd_loss.item() * inputs.size(0)
            running_total_loss += total_loss.item() * inputs.size(0)

        # (Classifer) Average Loss and Accuracy for the current epoch
        classifier_loss = running_cls_loss / dataset_sizes[phase]
        classifier_acc = running_corrects.double() / dataset_sizes[phase]

        # (VAE) Average Losses
        reconstruction_loss = running_recon_loss / dataset_sizes[phase]
        bce_loss = running_bce_loss / dataset_sizes[phase]
        kld_loss = running_kld_loss / dataset_sizes[phase]

        # (LSTD) Average Losses
        lstd_loss = running_lstd_loss / dataset_sizes[phase]

        # (Total) Average Loss over time
        total_loss = running_total_loss / dataset_sizes[phase]
        epoch_time = epoch_since - time.time()

        # Plot the Total loss, Reconstruction loss, LSTD loss and, Classification loss/accuracy
        # Send plots and images to tensorboard
        writer.add_scalar("Total Loss", total_loss, global_step=epoch)
        writer.add_scalar("Classification Loss", classifier_loss, global_step=epoch)
        writer.add_scalar("Classification Accuracy", classifier_acc, global_step=epoch)
        writer.add_scalar("VAE Loss", reconstruction_loss, global_step=epoch)
        writer.add_scalar("LSTD Loss", lstd_loss, global_step=epoch)

        if phase == 'train':
            linebrk = '='
            training_epoch_time = epoch_time
            train_log = (f'\nEpoch[{epoch + 1}/{epochs}] Total Loss: {total_loss:.3f}'
                         + f' | Classifer Loss: {classifier_loss:.3f} Acc: {classifier_acc:.3f}'
                         + f' | VAE Loss: {reconstruction_loss:.3f}'
                         # + f' | VAE Loss: {(reconstruction_loss):.3f} {(bce_loss):.3f} {(kld_loss):.3f}'
                         + f' | LSTD Loss: {lstd_loss:.3f}')
            print(f"{linebrk * 125}", train_log)
        else:
            epoch_time += training_epoch_time
            test_log = (f'{phase[0:3].capitalize()} Total Loss: {total_loss:.3f}'
                        + f' | Classifer Loss: {classifier_loss:.3f} Acc: {classifier_acc:.3f}'
                        + f' | VAE Loss: {reconstruction_loss:.3f}'
                        + f' | LSTD Loss: {lstd_loss:.3f}'
                        + f'\nTime:{epoch_time // 60:.0f}m {epoch_time % 60:.0f}s')
            print(test_log)

        # deep copy the model
    
        if phase == 'val' and classifier_acc > best_acc:
            print(f"Validation accuracy increased ({best_acc:.6f} --> {classifier_acc:.6f}).  Saving model ...")
            best_acc = classifier_acc
            needed_epochs = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())

            # For Reference
            # col_headers = ['time', 'accuracy', 'total_loss', 'classification_loss',
            #                'reconstruction_loss', 'lstd_loss', 'semantic similarity metric' , 'Lambda', 'classifier_architecture',
            #                'epochs', 'batch_size', 'optimizer', 'init_lr', 'scheduler']
            model_params = np.array([ts, best_acc.item(), total_loss, classifier_loss,
                                     reconstruction_loss, lstd_loss, args.metric , Lambda, model.classifier,
                                     needed_epochs, batch_size, optim_name, init_lr, scheduler_name], dtype=object)
            save_model(model, optim, criterion, needed_epochs, model_params, scheduler)


# Load, log and save the best model parameters
time_elapsed = time.time() - since
print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
print(f'Best Classification Acc: {best_acc:4f}')
model.load_state_dict(best_model_wts)
log_params(model_params, log_file, columns=col_headers)
save_model(model, optim, needed_epochs=needed_epochs, model_params=model.state_dict(), file_name=best_run_path)
writer.close()