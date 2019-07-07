### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.
import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.models as models
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import json
from PIL import ImageFile
import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
import os
from torchvision import datasets
ImageFile.LOAD_TRUNCATED_IMAGES = True

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
img_short_side_resize = 256
img_input_size = 224
shuffle = True
num_workers = 16
batch_size = 64

use_cuda = torch.cuda.is_available()
if not use_cuda:
    print('CUDA is not available.  Training on CPU ...')
    device = "cpu"
else:
    print('CUDA is available!  Training on GPU ...')
    device = torch.device("cuda:0")
    print("Using",torch.cuda.get_device_name(device))


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

model_transfer = models.resnet50(pretrained=True)

n_classes = 133

for param in model_transfer.parameters():
    param.requires_grad = False

# Freezing all parameters
for param in model_transfer.parameters():
    param.requires_grad = False

# Replacing the last layer (by default it will have requires_grad == True)
model_transfer.fc = nn.Linear(model_transfer.fc.in_features,n_classes)
# Initialize the weights of the new layer
nn.init.kaiming_normal_(model_transfer.fc.weight, nonlinearity='relu')
# Transfer to GPU
model_transfer = model_transfer.to(device)

model_transfer.eval()

model_transfer.load_state_dict(torch.load('model_transfer.pt'))

model_transfer.eval()

def dog_detector(img_path, model, device):
    ## TODO: Complete the function.
    out = model_predict(img_path,model,device)
    return out >= 151 and out <= 268 # true/false

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

alexnet = models.alexnet(pretrained=True)

def model_predict(img_path, model, device):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path

    Args:
        img_path: path to an image

    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    image = Image.open(img_path).convert('RGB')
    # To avoid changing the aspect ratio of the image we can use the FiveCrop transform, which returns the center and the
    # corner crops. Five outputs are calculated and then averaged.
    in_transform = transforms.Compose([
                        transforms.Resize(img_short_side_resize),
                        transforms.FiveCrop(img_input_size),
                        transforms.Lambda(lambda crops: torch.stack([transforms.Compose([
                    transforms.ToTensor(),transforms.Normalize(mean = norm_mean, std = norm_std)])(crop) for crop in crops]))])

    output = torch.argmax(model(in_transform(image).to(device)).mean(0))

    return output.to("cpu").item() # predicted class index

def process_probs(pred, probs, threshold = 0.01, max_n = 4):
    # Let's discard unlikely breeds
    selection = probs > threshold
    likely_breeds = [breed for (breed, detected) in zip(class_names, selection) if detected]
    likely_probs = [prob for (prob, detected) in zip(probs, selection) if detected]

    order = np.argsort(np.array(likely_probs))[::-1]
    likely_breeds = [likely_breeds[int(i)] for i in order]
    likely_probs = [likely_probs[int(i)] for i in order]
    likely_breeds = likely_breeds[:max_n]
    likely_probs = likely_probs[:max_n]

    # And let's fill the discarded gap with "other" (if there is a gap)
    if len(likely_breeds) > 1:
        likely_breeds = likely_breeds + ["Other"]
        likely_probs = likely_probs + [1 - sum(likely_probs)]
    else:
        # If no other breed is predicted with at least 1% let's round up the confidence to 100%
        likely_probs = [100]

    return likely_breeds, likely_probs


with open('dog_breeds.json') as json_file:
    class_names = json.load(json_file,strict=False)


def predict_breed_transfer(img_path, model, device):
    # load the image and return the predicted breed
    image = Image.open(img_path).convert('RGB')

    # Resnet replaces the fully connected layer with global average pooling. Theoretically then,
    # it should work with any input size, since the result of the global pooling depends only on
    # the number of filters and not on their spatial dimensions. Unfortunately, the provided
    # torchvision model implements global pooling as a regular average pooling in which the kernel
    # size is equal to the spatial size of the filters (in this case 7). So we still need to apply the
    # FiveCrop transform if we want to keep the aspect ratio and not only crop the center.

    in_transform = transforms.Compose([
                        transforms.Resize(img_short_side_resize),
                        transforms.FiveCrop(img_input_size),
                        transforms.Lambda(lambda crops: torch.stack([transforms.Compose([
                    transforms.ToTensor(),transforms.Normalize(mean = norm_mean, std = norm_std)])(crop) for crop in crops]))])

    scores = model(in_transform(image).to(device)).mean(0)
    output = torch.argmax(scores)

    return output.to("cpu").item(), F.softmax(scores,dim=0).to("cpu").data.numpy() # predicted class index

# First make sure we have all the models on the GPU
model_dog_detector = alexnet # Alexnet weirdly turned out to be the best dog detector
model_dog_detector = model_dog_detector.to(device)

def run_app(img_path):
    ## handle cases for a human face, dog, and neither

    # Check if a dog is detected
    dog_detected = dog_detector(img_path,model_dog_detector,device)
    # Check if a human is detected
    human_detected = face_detector(img_path)
    # Get the predicted breed(s)
    pred, probs = predict_breed_transfer(img_path, model_transfer, device)
    # Process class probabilities (remove very low probabilities and replace with "other", sort them etc)
    likely_breeds, likely_probs = process_probs(pred, probs)
    return likely_breeds, likely_probs, dog_detected, human_detected
