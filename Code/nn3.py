# Developer details: 
        # Name: Khushboo Mittal
        # Role: Architects
    # Version:
        # Version: V 1.0 (11 October 2024)
            # Developers: Khushboo Mittal
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet creates the Mask RCNN Model to train, evaluate, and predict whether 
    # a person is wearing a mask, not wearing a mask, or is wearing in an incorrect manner.

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Streamlit 1.36.0

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_box(obj):
    """Extracts bounding box coordinates from the XML object."""
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    return [xmin, ymin, xmax, ymax]

def generate_label(obj):
    """Assigns a label based on the object's name."""
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0

def generate_target(image_id, file): 
    """Creates a target dictionary with boxes and labels from the annotation file."""
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([image_id])
        target = {"boxes": boxes, "labels": labels, "image_id": img_id}
        return target

# Load image and label filenames
imgs = list(sorted(os.listdir("FaceMaskData/images/")))  
labels = list(sorted(os.listdir("FaceMaskData/annotations/"))) 

class MaskDataset(object):
    """Custom dataset class for loading images and annotations."""
    def __init__(self, transforms):
        self.transforms = transforms
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """Returns the image and its target (annotations) for a given index."""
        img_name = self.imgs[idx]
        label_name = self.labels[idx]

        img_path = os.path.join("FaceMaskData/images/", img_name)  
        label_path = os.path.join("FaceMaskData/annotations/", label_name)  
        img = Image.open(img_path).convert("RGB")
        target = generate_target(idx, label_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

data_transform = transforms.Compose([transforms.ToTensor()])

def collate_fn(batch):
    """Collates a batch of images and targets into a tuple."""
    return tuple(zip(*batch))

def get_model_instance_segmentation(num_classes):
    """Initializes a Faster R-CNN model for instance segmentation."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def save_model(model, model_path):
    """Saves the model state as a .pkl file."""
    with open(model_path, "wb") as f:
        pickle.dump(model.state_dict(), f)

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection / float(area_box1 + area_box2 - intersection) if (area_box1 + area_box2 - intersection) > 0 else 0
    return iou

def evaluate_predictions(preds, targets, iou_threshold=0.5):
    """Evaluates model predictions against target annotations using IoU."""
    true_positives, false_positives, false_negatives = 0, 0, 0
    correct_predictions = 0
    total_predictions = 0
    
    for pred, target in zip(preds, targets):
        pred_boxes = pred['boxes'].detach().cpu().numpy()
        pred_labels = pred['labels'].detach().cpu().numpy()
        target_boxes = target['boxes'].cpu().numpy()
        target_labels = target['labels'].cpu().numpy()

        matched = set()

        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1

            for j, target_box in enumerate(target_boxes):
                iou = calculate_iou(pred_box, target_box)
                if iou > best_iou and j not in matched:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and pred_labels[i] == target_labels[best_gt_idx]:
                true_positives += 1
                correct_predictions += 1
                matched.add(best_gt_idx)
            else:
                false_positives += 1
        
        false_negatives += len(target_boxes) - len(matched)
        total_predictions += len(pred_labels)
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return accuracy, precision, recall

def train_model(model_path):
    """Trains the model and saves the trained state."""
    model = get_model_instance_segmentation(3)  # Initialize the model with 3 classes
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load your dataset
    dataset = MaskDataset(data_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    # Training logic
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)
    model.to(device)
    
    num_epochs = 10  # Specify the number of epochs
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for imgs, annotations in data_loader:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model([imgs[0]], [annotations[0]])
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step() 
            epoch_loss += losses
            
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss}')
    
    # Save the model
    save_model(model, model_path)
    
    # Evaluate metrics
    model.eval()
    with torch.no_grad():
        preds = model(imgs)
    accuracy, precision, recall = evaluate_predictions(preds, annotations)

    return accuracy, precision, recall

# Example of calling the train_model function
# To train the model and save the metrics, use:
# metrics = train_model("your_model_path.pkl")
# print(metrics)

def load_model(model_path, num_classes):
    """Loads the trained model from the specified .pkl file."""
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(pickle.load(open(model_path, "rb")))
    model.eval()  # Set the model to evaluation mode
    return model

def visualize_predictions(model_path, image_path):
    """
    Visualizes the predicted bounding boxes and labels on the input image.
    
    Parameters:
    - model_path: Path to the trained model .pkl file.
    - image_path: Path to the input image.
    
    Returns:
    - predictions: A dictionary with predicted boxes, labels, and scores.
    """
    # Load the trained model
    model = load_model(model_path, num_classes=3)  # Adjust num_classes as needed

    # Load the image
    img = Image.open(image_path).convert("RGB")

    # Apply transformations
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Move to device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img_tensor = img_tensor.to(device)
    model.to(device)

    # Make predictions
    with torch.no_grad():
        preds = model(img_tensor)

    # Process predictions
    pred_boxes = preds[0]['boxes'].cpu().numpy()
    pred_scores = preds[0]['scores'].cpu().numpy()
    pred_labels = preds[0]['labels'].cpu().numpy()

    # Filter predictions based on a confidence threshold
    threshold = 0.5  # Set your confidence threshold
    boxes, labels, scores = [], [], []
    for i in range(len(pred_scores)):
        if pred_scores[i] > threshold:
            boxes.append(pred_boxes[i])
            labels.append(pred_labels[i])
            scores.append(pred_scores[i])  # Store scores for reference

    # Visualize the predictions
    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    ax = plt.gca()
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'Label: {label}, Score: {scores[i]:.2f}', fontsize=12, color='white', backgroundcolor='red')

    plt.axis('off')
    plt.show()

    # Return predictions
    return {'boxes': boxes, 'labels': labels, 'scores': scores}
