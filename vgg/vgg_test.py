# -*- coding: utf-8 -*-
from helper import *
from torch import nn
from time import strftime

import torch 
import torchvision.models as models

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

# Model + Hyper Paramaeters
vgg = models.vgg16(pretrained=False)
batch_size = 64
criterion = nn.CrossEntropyLoss()
vgg.name = 'VGG16'

#load model from checkpoint
checkpoint_path = "shrec_checkpoint.pth"

# Define devices and multi gpu if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   vgg = nn.DataParallel(vgg)

# Load the saved weights and parameters 
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

vgg = load_model(vgg, checkpoint_path , Display=True)

# load the testing dataset
shrec_sub_path = 'Data/Scene_SBR_IBR_Dataset/'
data_dir = Path(Path.home(), shrec_sub_path)
image_datasets, dataloader, dataset_sizes = load_data(data_dir,batch_size) 

vgg.cuda(device)
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

# Preform a test of the model on testing data
with torch.no_grad(): 
    for data in dataloader['test']:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward Pass
        output = vgg(inputs)

        # Class prediction and number of instance
        _, pred = torch.max(output.data, 1)
        top_prob, top_pred = torch.topk(output, 1, dim=1)
        # top_pred = top_pred.squeeze().tolist()
        # top_prob = top_prob.squeeze().tolist()
        top_prob, top_pred = top_prob.cpu().numpy(), top_pred.cpu().numpy()
        correct_label = np.append(correct_label, labels.cpu().numpy())
        correct_label_name = np.append(correct_label_name, np.array([classes[x] for x in np.nditer(labels.cpu().numpy())]))
        
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

# Save logged testing information to csv file
ts = strftime("%m.%d.%H:%M")
test_result_path =  'test_Results|'+ ts + '.csv'
test_result_path = Path.cwd() / test_result_path

print(f'{pred_label_name.shape}, {pred_labels.shape}, {pred_prob.shape}')
df = pd.DataFrame({'pred label': pred_labels.tolist(), 'ground truth label': correct_label.tolist(), 'ground truth label name': correct_label_name.tolist(), 'pred label name' : pred_label_name.tolist(), 'Prediction probability' : pred_prob.tolist()})
pd.DataFrame.to_csv(df, test_result_path)
