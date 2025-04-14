import torch
from tqdm import tqdm
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score)
import numpy as np
import wandb
from typing import Dict, Tuple, Any, Optional # Added for type hinting

# Try to import GoogLeNetOutputs if needed for type checking (adjust path if necessary)
try:
    # Assuming googlenet_custom.py is in the same directory or accessible
    from googlenet_custom import GoogLeNetOutputs
except ImportError:
    GoogLeNetOutputs = tuple # Fallback if the specific type isn't available

# Assuming config is available (adjust import if needed)
import config as CONFIGURE

class Trainer:
    """
    Handles the training and evaluation loops for a PyTorch model.

    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test/validation set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        device (torch.device): The device to run training/evaluation on (e.g., 'cuda', 'cpu').
    """
    def __init__(self, model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        # Store class names if available in config for confusion matrix labels
        self.class_names = getattr(CONFIGURE, 'class_names', [str(i) for i in range(getattr(CONFIGURE, 'num_classes', 1))])


    def _calculate_metrics(self, all_labels_np: np.ndarray,
                           all_predicted_np: np.ndarray,
                           all_outputs_np: np.ndarray) -> Dict[str, Any]:
        """
        Calculates various classification metrics.

        Args:
            all_labels_np: NumPy array of true labels.
            all_predicted_np: NumPy array of predicted labels.
            all_outputs_np: NumPy array of raw model outputs (logits).

        Returns:
            A dictionary containing calculated metrics.
        """
        metrics = {}

        # --- Standard Metrics ---
        metrics["f1_weighted"] = f1_score(y_true=all_labels_np, y_pred=all_predicted_np, average='weighted', zero_division=0)
        metrics["precision_weighted"] = precision_score(y_true=all_labels_np, y_pred=all_predicted_np, average='weighted', zero_division=0)
        metrics["recall_weighted"] = recall_score(y_true=all_labels_np, y_pred=all_predicted_np, average='weighted', zero_division=0)

        # --- ROC AUC ---
        # Calculate probabilities for ROC AUC
        try:
            # Softmax needs float tensor
            probs = torch.nn.functional.softmax(torch.from_numpy(all_outputs_np).float(), dim=1).numpy()
            # Ensure labels and probs have compatible shapes and types
            if probs.shape[1] == 1: # Binary case often outputs Nx1
                 probs = np.hstack([1-probs, probs]) # Make it Nx2

            if probs.shape[1] > 1 and len(np.unique(all_labels_np)) > 1:
                 metrics["roc_auc_weighted"] = roc_auc_score(all_labels_np, probs, multi_class='ovr', average='weighted')
            else:
                 # Handle cases where ROC AUC is not well-defined (e.g., single class in batch)
                 metrics["roc_auc_weighted"] = np.nan # Or 0.0 or None, depending on preference
                 print("Warning: ROC AUC calculation skipped (not enough classes or invalid shape).")

        except ValueError as e:
            print(f"Warning: Could not compute ROC AUC score: {e}")
            metrics["roc_auc_weighted"] = np.nan # Indicate failure

        # --- Confusion Matrix and Derived Metrics ---
        cm = confusion_matrix(all_labels_np, all_predicted_np)
        metrics["confusion_matrix"] = cm

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
            # Store per-class metrics if needed
            # metrics[f"sensitivity_class_{i}"] = sensitivity
            # metrics[f"specificity_class_{i}"] = specificity

        metrics["sensitivity_macro"] = np.mean(sensitivity_per_class) if sensitivity_per_class else 0
        metrics["specificity_macro"] = np.mean(specificity_per_class) if specificity_per_class else 0

        return metrics

    def train_epoch(self) -> Dict[str, float]:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_labels_list = []
        all_predicted_list = []
        all_outputs_list = [] # Store main logits

        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images = images.to(self.device).float()
            labels = labels.to(self.device).long()
            total_samples += labels.size(0)

            self.optimizer.zero_grad()

            # --- Get Model Output ---
            # Only pass images to the model
            model_output = self.model(images)

            # --- Handle GoogLeNet Output & Calculate Loss ---
            loss: torch.Tensor
            outputs_main: torch.Tensor

            # Check if aux_logits are active during training
            # hasattr check is safer if GoogLeNetOutputs isn't imported/available
            if self.model.training and getattr(self.model, 'aux_logits', False) and isinstance(model_output, tuple) and hasattr(model_output, 'logits'):
                outputs_main = model_output.logits
                outputs_aux1 = model_output.aux_logits1
                outputs_aux2 = model_output.aux_logits2

                loss_main = self.criterion(outputs_main, labels)
                loss_aux1 = self.criterion(outputs_aux1, labels)
                loss_aux2 = self.criterion(outputs_aux2, labels)
                # Standard weighting for GoogLeNet aux losses
                loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2
            else:
                # Evaluation mode, or aux_logits=False, or model doesn't return tuple
                outputs_main = model_output if isinstance(model_output, torch.Tensor) else model_output.logits # Ensure we get the tensor
                loss = self.criterion(outputs_main, labels)

            # --- Backpropagation ---
            loss.backward()
            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # --- Accumulate Loss and Predictions (use main logits) ---
            total_loss += loss.item() * labels.size(0) # Accumulate weighted loss
            _, predicted = torch.max(outputs_main.data, 1)
            correct_predictions += (predicted == labels).sum().item()

            # Store labels, predictions, and MAIN outputs for metrics calculation later
            all_labels_list.append(labels.cpu())
            all_predicted_list.append(predicted.cpu())
            all_outputs_list.append(outputs_main.detach().cpu()) # Detach before storing

        # --- Calculate Epoch Metrics ---
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        all_labels_tensor = torch.cat(all_labels_list, dim=0)
        all_predicted_tensor = torch.cat(all_predicted_list, dim=0)
        all_outputs_tensor = torch.cat(all_outputs_list, dim=0)

        # Calculate detailed metrics using the helper function
        epoch_metrics = self._calculate_metrics(
            all_labels_np=all_labels_tensor.numpy(),
            all_predicted_np=all_predicted_tensor.numpy(),
            all_outputs_np=all_outputs_tensor.numpy()
        )

        # Add basic metrics to the dictionary
        epoch_metrics['loss'] = avg_loss
        epoch_metrics['accuracy'] = accuracy

        return epoch_metrics

    def evaluate_epoch(self) -> Dict[str, float]:
        """Runs a single evaluation epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_labels_list = []
        all_predicted_list = []
        all_outputs_list = [] # Store main logits

        with torch.no_grad(): # Disable gradient calculations
            for images, labels in tqdm(self.test_loader, desc="Evaluating", leave=False):
                images = images.to(self.device).float()
                labels = labels.to(self.device).long()
                total_samples += labels.size(0)

                # --- Get Model Output ---
                # Only pass images to the model
                # In eval mode, GoogLeNet (even with aux_logits=True) should return only main logits
                outputs_main = self.model(images)
                # Ensure it's a tensor if the model unexpectedly returns a tuple in eval
                if not isinstance(outputs_main, torch.Tensor):
                     outputs_main = outputs_main.logits

                # --- Calculate Loss ---
                loss = self.criterion(outputs_main, labels)

                # --- Accumulate Loss and Predictions ---
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs_main.data, 1)
                correct_predictions += (predicted == labels).sum().item()

                # Store labels, predictions, and outputs
                all_labels_list.append(labels.cpu())
                all_predicted_list.append(predicted.cpu())
                all_outputs_list.append(outputs_main.cpu()) # No need to detach in no_grad context

        # --- Calculate Epoch Metrics ---
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        if not all_labels_list: # Handle empty loader case
             print("Warning: Evaluation loader was empty. Returning zero metrics.")
             return { 'loss': 0.0, 'accuracy': 0.0, 'f1_weighted': 0.0,
                      'precision_weighted': 0.0, 'recall_weighted': 0.0,
                      'roc_auc_weighted': 0.0, 'sensitivity_macro': 0.0,
                      'specificity_macro': 0.0, 'confusion_matrix': np.zeros((len(self.class_names), len(self.class_names)))}


        all_labels_tensor = torch.cat(all_labels_list, dim=0)
        all_predicted_tensor = torch.cat(all_predicted_list, dim=0)
        all_outputs_tensor = torch.cat(all_outputs_list, dim=0)

        # Calculate detailed metrics using the helper function
        epoch_metrics = self._calculate_metrics(
            all_labels_np=all_labels_tensor.numpy(),
            all_predicted_np=all_predicted_tensor.numpy(),
            all_outputs_np=all_outputs_tensor.numpy()
        )

        # Add basic metrics to the dictionary
        epoch_metrics['loss'] = avg_loss
        epoch_metrics['accuracy'] = accuracy

        return epoch_metrics

    def train(self, num_epochs: int):
        """Runs the full training loop for a specified number of epochs."""

        # Initialize dictionaries to store best metrics
        best_metrics = {
            "train": {"loss": float('inf'), "accuracy": 0.0, "f1_weighted": 0.0, "roc_auc_weighted": 0.0},
            "val": {"loss": float('inf'), "accuracy": 0.0, "f1_weighted": 0.0, "roc_auc_weighted": 0.0}
            # Add other metrics if you want to track their best values specifically
        }
        best_val_acc_epoch = -1 # Track epoch of best validation accuracy

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

            train_epoch_metrics = self.train_epoch()
            val_epoch_metrics = self.evaluate_epoch() # Use test_loader as validation

            # --- Logging ---
            if CONFIGURE.wandb:
                log_data = {}
                # Add train metrics with prefix
                for key, value in train_epoch_metrics.items():
                    if key != "confusion_matrix": # Don't log raw CM array directly this way
                        log_data[f"train/{key}"] = value
                    else:
                         # Log confusion matrix as a W&B table
                         try:
                             cm_table = wandb.Table(data=value,
                                                   columns=self.class_names,
                                                   rows=self.class_names)
                             log_data[f"train/confusion_matrix"] = cm_table
                         except Exception as e:
                              print(f"Could not log train confusion matrix to W&B: {e}")


                # Add validation metrics with prefix
                for key, value in val_epoch_metrics.items():
                    if key != "confusion_matrix":
                         log_data[f"val/{key}"] = value
                    else:
                         # Log confusion matrix as a W&B table
                         try:
                            cm_table = wandb.Table(data=value,
                                                   columns=self.class_names,
                                                   rows=self.class_names)
                            log_data[f"val/confusion_matrix"] = cm_table
                         except Exception as e:
                            print(f"Could not log validation confusion matrix to W&B: {e}")


                log_data["epoch"] = epoch + 1 # Log current epoch (1-based)
                wandb.log(log_data)

            # --- Print Metrics ---
            print("--- Training Metrics ---")
            for key, value in train_epoch_metrics.items():
                if key != "confusion_matrix":
                    print(f"{key.replace('_', ' ').capitalize()}: {value:.4f}")
                else:
                    print(f"{key.replace('_', ' ').capitalize()}:\n{value}")

            print("\n--- Validation Metrics ---")
            for key, value in val_epoch_metrics.items():
                 if key != "confusion_matrix":
                    print(f"{key.replace('_', ' ').capitalize()}: {value:.4f}")
                 else:
                    print(f"{key.replace('_', ' ').capitalize()}:\n{value}")


            # --- Update Best Metrics (Example: based on validation accuracy) ---
            # You might choose validation loss, F1, or another metric
            current_val_acc = val_epoch_metrics.get('accuracy', 0.0)
            if current_val_acc > best_metrics["val"]["accuracy"]:
                print(f"\n>>> New best validation accuracy: {current_val_acc:.4f} (Previous: {best_metrics['val']['accuracy']:.4f})")
                best_metrics["val"]["accuracy"] = current_val_acc
                best_val_acc_epoch = epoch + 1
                # Update all best validation metrics when primary improves
                for key in best_metrics["val"]:
                     if key in val_epoch_metrics:
                           best_metrics["val"][key] = val_epoch_metrics[key]

                # TODO: Save model checkpoint here based on best validation metric
                # Example: torch.save(self.model.state_dict(), f"best_model_epoch_{epoch+1}.pth")


            # Update best training metrics (less common to track, but included)
            current_train_acc = train_epoch_metrics.get('accuracy', 0.0)
            if current_train_acc > best_metrics["train"]["accuracy"]:
                 best_metrics["train"]["accuracy"] = current_train_acc
            # ... update other best train metrics if needed ...


        # --- End of Training ---
        print("\n--- Training Finished ---")
        print(f"Best Validation Accuracy: {best_metrics['val']['accuracy']:.4f} at Epoch {best_val_acc_epoch}")

        print("\nBest Recorded Validation Metrics (at best accuracy epoch):")
        for key, value in best_metrics["val"].items():
            print(f"Best val {key.replace('_', ' ').capitalize()}: {value:.4f}")

        # Optionally print best recorded training metrics
        print("\nBest Recorded Training Metrics:")
        for key, value in best_metrics["train"].items():
           print(f"Best train {key.replace('_', ' ')}: {value:.4f}")

        print("\nTraining complete.")