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

# Import necessary libraries
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from bs4 import BeautifulSoup
import torch
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.patches as patches
import os
import pickle  
from pymongo import MongoClient
# from db_utils import load_FaceMask_data_from_mongodb as load_data_from_mongodb

# Generate bounding boxes and labels
def generate_box(obj):
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    return [xmin, ymin, xmax, ymax]

def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0

def generate_target(image_id, file): 
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        boxes = [generate_box(obj) for obj in objects]
        labels = [generate_label(obj) for obj in objects]
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([image_id])
        
        return {"boxes": boxes, "labels": labels, "image_id": img_id}

# Custom dataset for Mask RCNN
class MaskDataset(object):
    def __init__(self, db, collection_name, transforms=None):
        self.transforms = transforms
        self.img_paths, self.label_paths = load_data_from_mongodb(db, collection_name)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]

        img = Image.open(img_path).convert("RGB")
        target = generate_target(idx, label_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

# Data transformation
data_transform = transforms.Compose([transforms.ToTensor()])

# Model architecture modification
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Train the model
def train_model_nn(mongodb_host, mongodb_port, mongodb_db,collection_name, model_path, num_epochs=10):
    # Connect to MongoDB
    client = MongoClient(host=mongodb_host, port=mongodb_port)
    db = client[mongodb_db]

    # Create dataset and data loader
    dataset = MaskDataset(db,collection_name, data_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=lambda x: tuple(zip(*x)))

    # Define the model and optimizer
    model = get_model_instance_segmentation(3)  # 3 classes: no mask, with mask, incorrect mask
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for imgs, annotations in data_loader:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        # Evaluate predictions to calculate accuracy
        model.eval()
        all_preds = []
        all_annotations = []
        
        with torch.no_grad():
            for imgs, annotations in data_loader:
                imgs = list(img.to(device) for img in imgs)
                annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
                preds = model(imgs)  # Get predictions
                all_preds.extend(preds)  # Collect all predictions
                all_annotations.extend(annotations)  # Collect all annotations

        metrics = evaluate_predictions(all_preds, all_annotations)  # Evaluate predictions
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1_score']

    # Save the trained model
    save_model(model, model_path)
    return accuracy, precision, recall, f1

# Save model function
def save_model(model, model_path):
    with open(model_path, "wb") as f:
        pickle.dump(model.state_dict(), f)

# Load model function
def load_model(model_path, num_classes=3):
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

def calculate_iou(box1, box2):
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def evaluate_predictions(preds, targets, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    ious = []
    
    for pred, target in zip(preds, targets):
        pred_boxes = pred['boxes'].detach().cpu().numpy()
        pred_labels = pred['labels'].detach().cpu().numpy()
        
        target_boxes = target['boxes'].cpu().numpy()
        target_labels = target['labels'].cpu().numpy()
        
        matched = set()  # Track matched ground truth boxes
        
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
                matched.add(best_gt_idx)
                ious.append(best_iou)
            else:
                false_positives += 1
        
        false_negatives += len(target_boxes) - len(matched)
    # Calculate metrics
    total_predictions = true_positives + false_positives + false_negatives
    accuracy = true_positives / total_predictions if total_predictions > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # mean_iou = np.mean(ious) if ious else 0
    
    return accuracy, precision, recall, f1_score

# # Evaluate model on test data
# model.eval()
# with torch.no_grad():
#     preds = model(imgs)  # Predictions

# # Call evaluation function
# metrics = evaluate_predictions(preds, annotations)
# print(f"Precision: {metrics['precision']:.4f}")
# print(f"Recall: {metrics['recall']:.4f}")
# print(f"Mean IoU: {metrics['mean_iou']:.4f}")

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# Class label mapping
class_mapping = {0: "Without Mask", 1: "With Mask", 2: "Mask Worn Incorrectly"}

def plot_image(image, prediction):
    # Convert image tensor to a NumPy array
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy().transpose(1, 2, 0)  # Convert to (H, W, C)
    
    # Plot the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot bounding boxes
    boxes = prediction['boxes'].detach().cpu().numpy()  # Detach boxes from computation graph
    labels = prediction['labels'].detach().cpu().numpy()  # Detach labels from computation graph

    # Plot each box with class names
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, class_mapping[labels[i]], color='white', fontsize=12, 
                bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')
    plt.show()

# # Display predictions
# for img, pred in zip(imgs, preds):
#     plot_image(img, pred)

# # Save the model as .pkl file
# model_save_path = "mask_rcnn_model.pkl"  # Specifies the filename

# # Open the file in write-binary mode and save the model's state dictionary
# with open(model_save_path, "wb") as f:
#     pickle.dump(model.state_dict(), f)


# def load_model():
#     model = get_model_instance_segmentation(3)
#     model.load_state_dict(torch.load(model_save_path))
#     return model

# To load the model later, simply call:
# model = load_model()
