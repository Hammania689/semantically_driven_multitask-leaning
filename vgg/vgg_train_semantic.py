from torch import nn
from helper import *
from tensorboardX import SummaryWriter

import torch 
import torchvision.models as models

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

# Model + Hyper Paramaeters
vgg = models.vgg16(pretrained=False)



# Change classifier accordingly
vgg.classifier = nn.Sequential(
    nn.Linear(in_features=25088, out_features=4096, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=4096, out_features=365, bias=True)
)

# Load in converted weights into the untrained model 
places365_weights = torch.load('vgg_places365.pth')
vgg.load_state_dict(places365_weights)

# Freeze all layers in VGG
for param in vgg.parameters():
    param.requires_grad = False

# Change classifier accordingly
vgg.classifier = nn.Sequential(
    nn.Linear(in_features=25088, out_features=4096, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=4096, out_features=1028, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=1028, out_features=512, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=512, out_features=30, bias=True)
)

lr = 0.001
optim = torch.optim.SGD(vgg.parameters(),lr=lr, momentum=0.9)
epochs = 50
batch_size = 64
criterion = nn.CrossEntropyLoss() 
vgg.name = 'VGG16'

#load model from checkpoint
checkpoint_path = Path.cwd() /  'notable_runs' / 'shrec_checkpoint.pth'
lesk_path = Path.cwd() / 'Comprehensive_Lesk_Similary_Distances.csv'
path_to_yolo_outputs = Path.cwd() / 'XXXXXXXXXXXXcount_result_30classes'

lesk_distances = pd.read_csv(lesk_path)
visualized_ground_truth = yolo_output_to_dict(path_to_yolo_outputs)

# Define devices and multi gpu if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# vgg = load_model(vgg, checkpoint_path / 'checkpoint.pth', Display=True)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  vgg = nn.DataParallel(vgg)
vgg.to(device)

# Load places data for cross validation
shrec_sub_path = 'Data/Scene_SBR_IBR_Dataset/'
data_dir = Path(Path.home(), shrec_sub_path)
dataset, dataloader, dataset_sizes = load_data(data_dir, batch_size=batch_size)


writer = SummaryWriter()
phases = ['train', 'val']
since = time.time()
best_model_wts = 0
best_acc = 0.0

needed_epochs = 0
training_epoch_time = 0


# Semantic BCE loss
def semantic_bce(vgg_pred, labels):
    # Batch similarity distance
    similarity_dist = torch.zeros((batch_size, 210))
    summed_vis_truth = torch.zeros((batch_size, 1, 210))
    
    # For each label in the Current Batch
    for label in range(len(labels)):
        # Get label name  
        # Update the sliced similarity accordingly 
        cur_label_name = labels[label]
        similarity_dist[label] = torch.from_numpy(lesk_distances[classes[cur_label_name]].values)
        summed_vis_truth[label] = visualized_ground_truth[classes[cur_label_name]]

    LAMBDA = 0.5
    CME = torch.dot(summed_vis_truth[label], similarity_dist)
    loss = F.binary_cross_entropy(summed_vis_truth, vgg_pred, reduction='mean') + LAMBDA * F.binary_cross_entropy(CME, vgg_pred.max(dim=1) ,reduction='mean')
    return loss


phase = ['train', 'val']
for epoch in range(epochs):
    for phase in phases:
        if phase == 'train':
            vgg.train()  # Set model to training mode
            vgg.to(device)
        else:
            vgg.eval()  # Set model to evaluate mode
            vgg.to(device)

        running_cls_loss = 0.0
        running_corrects = 0

        epoch_since = time.time()

        for idx, (inputs, labels) in enumerate(dataloader[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optim.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                # Forward Pass
                output = vgg(inputs)

                # Class prediction and number of instance
                _, pred = torch.max(output.data, 1)
                loss = semantic_bce(output,labels)

                # Calculate loss and Backprop
                if phase == 'train':
                    # NOTE: this needs to be done for the current implementation. Can only calculate gradient on single value 
                    loss.sum().backward() 
                    optim.step()

            # (Classifer) Average Loss and Accuracy for current batch
            running_cls_loss += loss.sum().item() * inputs.size(0)
            running_corrects += torch.sum(pred == labels.data)

            # if idx % 10000 == 0:
            # print(f'Batch Loss: {running_cls_loss}| Acc: {running_corrects / batch_size}')
        
        # (Classifer) Average Loss and Accuracy for the current epoch
        classifier_loss = running_cls_loss / dataset_sizes[phase]
        classifier_acc = running_corrects.double() / dataset_sizes[phase]

        epoch_time = epoch_since - time.time()

        writer.add_scalar("Classification Loss", classifier_loss, global_step=epoch)
        writer.add_scalar("Classification Accuracy", classifier_acc, global_step=epoch)
        # writer.add_graph(models.vgg16.cuda(), (inputs, labels), global_step=epoch)

        if phase == 'train':
            linebrk = '='
            training_epoch_time = epoch_time
            train_log = (f'\nEpoch[{epoch + 1}/{epochs}]\n'
                            + f'Training | Classifer Loss: {classifier_loss:.3f} Acc: {classifier_acc:.3f}')
            print(f"{linebrk * 125}", train_log)
        else:
            epoch_time += training_epoch_time
            test_log = (f'\n Validation'
                            + f' | Classifer Loss: {classifier_loss:.3f} Acc: {classifier_acc:.3f}'
                            + f'\nTime:{epoch_time // 60:.0f}m {epoch_time % 60:.0f}s')
            print(test_log)

        # deep copy the model
        if phase == 'val' and classifier_acc > best_acc:
            print(f"Validation accuracy increased ({best_acc:.6f} --> {classifier_acc:.6f}).  Saving model ...")
            best_acc = classifier_acc
            needed_epochs = epoch + 1
            best_model_wts = copy.deepcopy(vgg.state_dict())
            save_model(vgg, optim, criterion, needed_epochs)

time_elapsed = time.time() - since
print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
print(f'Best Classification Acc: {best_acc:4f}')


# load best model weights
vgg.module.load_state_dict(best_model_wts)
save_model(vgg, optim, criterion, needed_epochs)
writer.close()