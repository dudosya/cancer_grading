import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import numpy as np
import wandb
import config as CONFIGURE

class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device):
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
        all_outputs = []

        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images = images.to(self.device).float()
            labels = labels.to(self.device).long()

            self.optimizer.zero_grad()
            outputs, features = self.model(images, labels)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.append(labels.cpu())
            all_predicted.append(predicted.cpu())
            all_outputs.append(outputs.detach().cpu())

        avg_loss = total_loss / total
        accuracy = correct / total

        all_labels_tensor = torch.cat(all_labels, dim=0)
        all_predicted_tensor = torch.cat(all_predicted, dim=0)
        all_outputs_tensor = torch.cat(all_outputs, dim=0)

        labels_np = all_labels_tensor.numpy()
        predicted_np = all_predicted_tensor.numpy()
        outputs_np = all_outputs_tensor.numpy()
        probs = torch.nn.functional.softmax(torch.tensor(outputs_np), dim=1).numpy()

        epoch_f1 = f1_score(y_true=labels_np, y_pred=predicted_np, average='weighted', zero_division=0)
        epoch_precision = precision_score(y_true=labels_np, y_pred=predicted_np, average='weighted', zero_division=0)
        epoch_recall = recall_score(y_true=labels_np, y_pred=predicted_np, average='weighted', zero_division=0)

        # Compute ROC AUC
        try:
            roc_auc = roc_auc_score(labels_np, probs, multi_class='ovr', average='weighted')
        except ValueError:
            roc_auc = np.nan  # If computation fails due to single-class labels in this epoch

        # Compute confusion matrix
        cm = confusion_matrix(labels_np, predicted_np)
        print("Training Confusion Matrix")
        print(cm)
        print(" ")

        # Compute sensitivity and specificity per class
        n_classes = cm.shape[0]
        sensitivity_per_class = []
        specificity_per_class = []

        for i in range(n_classes):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - (TP + FN + FP)

            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

            sensitivity_per_class.append(sensitivity)
            specificity_per_class.append(specificity)

        avg_sensitivity = np.mean(sensitivity_per_class)
        avg_specificity = np.mean(specificity_per_class)

        return (avg_loss, accuracy, epoch_f1, epoch_precision, epoch_recall,
                avg_sensitivity, avg_specificity, roc_auc)

    def evaluate_epoch(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_predicted = []
        all_outputs = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating", leave=False):
                images = images.to(self.device).float()
                labels = labels.to(self.device).long()

                outputs, features = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.append(labels.cpu())
                all_predicted.append(predicted.cpu())
                all_outputs.append(outputs.cpu())

            all_labels_tensor = torch.cat(all_labels, dim=0)
            all_predicted_tensor = torch.cat(all_predicted, dim=0)
            all_outputs_tensor = torch.cat(all_outputs, dim=0)

            avg_loss = total_loss / total
            accuracy = correct / total
            labels_np = all_labels_tensor.numpy()
            predicted_np = all_predicted_tensor.numpy()
            outputs_np = all_outputs_tensor.numpy()
            probs = torch.nn.functional.softmax(torch.tensor(outputs_np), dim=1).numpy()

            epoch_f1 = f1_score(y_true=labels_np, y_pred=predicted_np, average='weighted', zero_division=0)
            epoch_precision = precision_score(y_true=labels_np, y_pred=predicted_np, average='weighted', zero_division=0)
            epoch_recall = recall_score(y_true=labels_np, y_pred=predicted_np, average='weighted', zero_division=0)

            # Compute ROC AUC
            try:
                roc_auc = roc_auc_score(labels_np, probs, multi_class='ovr', average='weighted')
            except ValueError:
                roc_auc = np.nan  # If computation fails due to single-class labels in this epoch

            # Compute confusion matrix
            cm = confusion_matrix(labels_np, predicted_np)
            print("Test Confusion Matrix")
            print(cm)
            print(" ")

            # Compute sensitivity and specificity per class
            n_classes = cm.shape[0]
            sensitivity_per_class = []
            specificity_per_class = []

            for i in range(n_classes):
                TP = cm[i, i]
                FN = cm[i, :].sum() - TP
                FP = cm[:, i].sum() - TP
                TN = cm.sum() - (TP + FN + FP)

                sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

                sensitivity_per_class.append(sensitivity)
                specificity_per_class.append(specificity)

            avg_sensitivity = np.mean(sensitivity_per_class)
            avg_specificity = np.mean(specificity_per_class)

            return (avg_loss, accuracy, epoch_f1, epoch_precision, epoch_recall,
                    avg_sensitivity, avg_specificity, roc_auc)

    def train(self, num_epochs):
        # Initialize best metrics for training
        best_train_loss = float('inf')
        best_train_acc = 0.0
        best_train_f1 = 0.0
        best_train_precision = 0.0
        best_train_recall = 0.0
        best_train_sensitivity = 0.0
        best_train_specificity = 0.0
        best_train_roc_auc = 0.0

        # Initialize best metrics for testing
        best_test_loss = float('inf')
        best_test_acc = 0.0
        best_test_f1 = 0.0
        best_test_precision = 0.0
        best_test_recall = 0.0
        best_test_sensitivity = 0.0
        best_test_specificity = 0.0
        best_test_roc_auc = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

            train_metrics = self.train_epoch()
            test_metrics = self.evaluate_epoch()

            (train_loss, train_acc, train_f1, train_precision, train_recall, train_sensitivity, train_specificity, train_roc_auc) = train_metrics

            (test_loss, test_acc, test_f1, test_precision, test_recall, test_sensitivity, test_specificity, test_roc_auc) = test_metrics

            if CONFIGURE.wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "train_sensitivity": train_sensitivity,
                    "train_specificity": train_specificity,
                    "train_roc_auc": train_roc_auc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_f1": test_f1,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_sensitivity": test_sensitivity,
                    "test_specificity": test_specificity,
                    "test_roc_auc": test_roc_auc,
                    "epoch": epoch
                })

            print("TRAINING METRICS")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Train F1: {train_f1:.4f}")
            print(f"Train Precision: {train_precision:.4f}")
            print(f"Train Recall: {train_recall:.4f}")
            print(f"Train Sensitivity: {train_sensitivity:.4f}")
            print(f"Train Specificity: {train_specificity:.4f}")
            print(f"Train ROC AUC: {train_roc_auc:.4f}")

            print("\nTEST METRICS")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test F1: {test_f1:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            print(f"Test Sensitivity: {test_sensitivity:.4f}")
            print(f"Test Specificity: {test_specificity:.4f}")
            print(f"Test ROC AUC: {test_roc_auc:.4f}")

            #TODO: this could have been written much better tbh
            # Update best training metrics
            if train_loss < best_train_loss:
                best_train_loss = train_loss
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            if train_f1 > best_train_f1:
                best_train_f1 = train_f1
            if train_precision > best_train_precision:
                best_train_precision = train_precision
            if train_recall > best_train_recall:
                best_train_recall = train_recall
            if train_sensitivity > best_train_sensitivity:
                best_train_sensitivity = train_sensitivity
            if train_specificity > best_train_specificity:
                best_train_specificity = train_specificity
            if train_roc_auc > best_train_roc_auc:
                best_train_roc_auc = train_roc_auc

            # Update best testing metrics
            if test_loss < best_test_loss:
                best_test_loss = test_loss
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
            if test_precision > best_test_precision:
                best_test_precision = test_precision
            if test_recall > best_test_recall:
                best_test_recall = test_recall
            if test_sensitivity > best_test_sensitivity:
                best_test_sensitivity = test_sensitivity
            if test_specificity > best_test_specificity:
                best_test_specificity = test_specificity
            if test_roc_auc > best_test_roc_auc:
                best_test_roc_auc = test_roc_auc

        # After training loop, print best metrics
        print("\nBest Training Metrics:")
        print(f"Best Train Loss: {best_train_loss:.4f}")
        print(f"Best Train Accuracy: {best_train_acc:.4f}")
        print(f"Best Train F1: {best_train_f1:.4f}")
        print(f"Best Train Precision: {best_train_precision:.4f}")
        print(f"Best Train Recall: {best_train_recall:.4f}")
        print(f"Best Train Sensitivity: {best_train_sensitivity:.4f}")
        print(f"Best Train Specificity: {best_train_specificity:.4f}")
        print(f"Best Train ROC AUC: {best_train_roc_auc:.4f}")

        print("\nBest Testing Metrics:")
        print(f"Best Test Loss: {best_test_loss:.4f}")
        print(f"Best Test Accuracy: {best_test_acc:.4f}")
        print(f"Best Test F1: {best_test_f1:.4f}")
        print(f"Best Test Precision: {best_test_precision:.4f}")
        print(f"Best Test Recall: {best_test_recall:.4f}")
        print(f"Best Test Sensitivity: {best_test_sensitivity:.4f}")
        print(f"Best Test Specificity: {best_test_specificity:.4f}")
        print(f"Best Test ROC AUC: {best_test_roc_auc:.4f}")
