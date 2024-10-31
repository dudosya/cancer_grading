import torch
from tqdm import tqdm
from sklearn.metrics import f1_score,precision_score, recall_score, confusion_matrix
from collections import Counter
import numpy as np

class Trainer:
    def __init__(self,model,train_loader, test_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_predicted = []
        
        for images,labels in tqdm(self.train_loader, desc="Training", leave=False):
            images = images.to(self.device).float()
            labels = labels.to(self.device).long()
            
            self.optimizer.zero_grad()
            outputs, features = self.model(images,labels)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.append(labels)
            all_predicted.append(predicted)
            
        avg_loss = total_loss / total
        accuracy = correct / total
        
        all_labels_tensor = torch.cat(all_labels, dim=0)
        all_predicted_tensor = torch.cat(all_predicted, dim=0)
        
        labels_np = all_labels_tensor.cpu().numpy()
        predicted_np = all_predicted_tensor.cpu().numpy()
        epoch_f1 = f1_score(y_true=labels_np, y_pred=predicted_np, average='weighted', zero_division=0)
        epoch_precision = precision_score(y_true=labels_np, y_pred=predicted_np, average='weighted', zero_division=0)
        epoch_recall = recall_score(y_true=labels_np, y_pred=predicted_np, average='weighted', zero_division=0)
        
        print(confusion_matrix(y_true=labels_np, y_pred=predicted_np))
        
        return avg_loss, accuracy, epoch_f1, epoch_precision, epoch_recall
    
    def evaluate_epoch(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_predicted = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating", leave=False):
                images = images.to(self.device).float()
                labels = labels.to(self.device).long()
                
                outputs, features = self.model(images)
                loss = self.criterion(outputs,labels)
                
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.append(labels)
                all_predicted.append(predicted)
            
            
            all_labels_tensor = torch.cat(all_labels, dim=0)
            all_predicted_tensor = torch.cat(all_predicted, dim=0)
            
            avg_loss = total_loss / total
            accuracy = correct/total
            labels_np = all_labels_tensor.cpu().numpy()
            predicted_np = all_predicted_tensor.cpu().numpy()
            epoch_f1 = f1_score(y_true=labels_np, y_pred=predicted_np, average='weighted', zero_division=0)
            epoch_precision = precision_score(y_true=labels_np, y_pred=predicted_np, average='weighted', zero_division=0)
            epoch_recall = recall_score(y_true=labels_np, y_pred=predicted_np, average='weighted', zero_division=0)
        
            print(confusion_matrix(y_true=labels_np, y_pred=predicted_np))
            
            return avg_loss, accuracy, epoch_f1, epoch_precision, epoch_recall
        
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            
            train_loss, train_acc, train_f1, train_precision, train_recall = self.train_epoch()
            test_loss, test_acc, test_f1, test_precision, test_recall = self.evaluate_epoch()
            
            print("TRAINING METRICS")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Train F1: {train_f1:.4f}")
            print(f"Train Precision: {train_precision:.4f}")
            print(f"Train Recall: {train_recall:.4f}")
            
            print("\nTEST METRICS")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test F1: {test_f1:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall: {test_recall:.4f}")