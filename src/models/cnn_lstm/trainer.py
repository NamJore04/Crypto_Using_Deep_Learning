"""
Training pipeline for CNN-LSTM model.

This module provides training, validation, and evaluation functionality
for the CNN-LSTM model with early stopping and performance monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .model import CNNLSTMModel, CNNLSTMModelV2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import ModelConfig

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Training pipeline for CNN-LSTM model.
    
    Handles model training, validation, evaluation, and performance monitoring
    with early stopping and learning rate scheduling.
    """
    
    def __init__(self, model: nn.Module, config: ModelConfig, device: Optional[str] = None):
        """
        Initialize model trainer.
        
        Args:
            model: CNN-LSTM model to train
            config: Model configuration settings
            device: Device to use for training ('cuda', 'cpu', or None for auto)
        """
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def setup_training(self, learning_rate: Optional[float] = None, 
                      weight_decay: float = 1e-5) -> None:
        """
        Setup optimizer and scheduler for training.
        
        Args:
            learning_rate: Learning rate (uses config default if None)
            weight_decay: Weight decay for regularization
        """
        if learning_rate is None:
            learning_rate = self.config.learning_rate
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        logger.info(f"Setup training with lr={learning_rate}, weight_decay={weight_decay}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train model for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple[float, float]: Average loss and accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple[float, float]: Average loss and accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: Optional[int] = None, early_stopping_patience: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (uses config default if None)
            early_stopping_patience: Early stopping patience (uses config default if None)
            
        Returns:
            Dict[str, Any]: Training results and history
        """
        if epochs is None:
            epochs = self.config.epochs
        if early_stopping_patience is None:
            early_stopping_patience = self.config.early_stopping_patience
        
        logger.info(f"Starting training for {epochs} epochs with patience {early_stopping_patience}")
        
        # Reset training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Training loop
        for epoch in range(epochs):
            # Train epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate epoch
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Log progress
            logger.info(
                f'Epoch {epoch+1}/{epochs}: '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
            )
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_acc
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= early_stopping_patience:
                    logger.info(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f'Loaded best model with val_loss: {self.best_val_loss:.4f}')
        
        # Return training results
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'epochs_trained': len(self.train_losses)
        }
    
    def evaluate(self, test_loader: DataLoader, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            class_names: Names of classes for reporting
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        if class_names is None:
            class_names = ['SIDEWAY', 'UPTREND', 'DOWNTREND', 'BREAKOUT']
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_macro = f1_score(all_targets, all_predictions, average='macro')
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        
        # Per-class metrics
        report = classification_report(
            all_targets, all_predictions,
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Per-class F1 scores
        f1_per_class = {}
        for i, class_name in enumerate(class_names):
            f1_per_class[class_name] = f1_score(
                all_targets, all_predictions, 
                labels=[i], average='binary', zero_division=0
            )
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probabilities)
        }
        
        logger.info(f'Test Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}')
        
        return results
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.train_accuracies, label='Train Accuracy')
        axes[0, 1].plot(self.val_accuracies, label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.learning_rates)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Loss difference plot
        loss_diff = [abs(t - v) for t, v in zip(self.train_losses, self.val_losses)]
        axes[1, 1].plot(loss_diff)
        axes[1, 1].set_title('Train-Validation Loss Difference')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: Names of classes
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        
        logger.info(f"Model loaded from {filepath}")


def create_data_loaders(X: np.ndarray, y: np.ndarray, config: ModelConfig,
                       train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        X: Feature data
        y: Target data
        config: Model configuration
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test loaders
    """
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )
    
    logger.info(f"Created data loaders: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
