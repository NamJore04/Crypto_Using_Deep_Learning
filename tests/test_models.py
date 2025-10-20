"""
Unit tests for model components.

Tests CNN-LSTM, CNN-Transformer, and ensemble models.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.cnn_lstm.model import CNNLSTMModel, CNNLSTMModelV2
from models.transformer.model import CNNTransformerModel, CNNTransformerModelV2, PositionalEncoding
from models.ensemble.model_ensemble import ModelEnsemble, AdaptiveEnsemble
from models.cnn_lstm.trainer import ModelTrainer, create_data_loaders
from config.settings import ModelConfig


class TestCNNLSTMModel:
    """Test cases for CNN-LSTM model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 20
        self.sequence_length = 60
        self.hidden_dim = 64
        self.num_layers = 2
        self.num_classes = 4
        
        # Create ModelConfig object
        self.config = ModelConfig(
            input_dim=self.input_dim,
            sequence_length=self.sequence_length,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            dropout_rate=0.3
        )
        
        self.model = CNNLSTMModel(self.config)
    
    def test_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert hasattr(self.model, 'conv1')
        assert hasattr(self.model, 'lstm')
        assert hasattr(self.model, 'classifier')
    
    def test_forward_pass(self):
        """Test forward pass through model."""
        batch_size = 2
        # Fix input shape: models expect (batch, sequence, features)
        input_tensor = torch.randn(batch_size, self.sequence_length, self.input_dim)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        assert output.shape == (batch_size, self.num_classes)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_parameters(self):
        """Test model parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params
    
    def test_model_training_mode(self):
        """Test model training and evaluation modes."""
        # Training mode
        self.model.train()
        assert self.model.training
        
        # Evaluation mode
        self.model.eval()
        assert not self.model.training


class TestCNNLSTMModelV2:
    """Test cases for CNN-LSTM model V2 with residual connections."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 20
        self.sequence_length = 60
        self.hidden_dim = 64
        self.num_layers = 2
        self.num_classes = 4
        
        # Create ModelConfig object
        self.config = ModelConfig(
            input_dim=self.input_dim,
            sequence_length=self.sequence_length,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            dropout_rate=0.3
        )
        
        self.model = CNNLSTMModelV2(self.config)
    
    def test_initialization(self):
        """Test model V2 initialization."""
        assert self.model is not None
        assert hasattr(self.model, 'conv1')
        assert hasattr(self.model, 'lstm')
        assert hasattr(self.model, 'classifier')
    
    def test_forward_pass(self):
        """Test forward pass through model V2."""
        batch_size = 2
        # Fix input shape: models expect (batch, sequence, features)
        input_tensor = torch.randn(batch_size, self.sequence_length, self.input_dim)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        assert output.shape == (batch_size, self.num_classes)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_residual_connections(self):
        """Test residual connections work correctly."""
        batch_size = 2
        # Fix input shape: models expect (batch, sequence, features)
        input_tensor = torch.randn(batch_size, self.sequence_length, self.input_dim)
        
        # Test that residual connections don't cause dimension mismatches
        with torch.no_grad():
            output = self.model(input_tensor)
        
        assert output.shape == (batch_size, self.num_classes)


class TestCNNTransformerModel:
    """Test cases for CNN-Transformer model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 20
        self.sequence_length = 60
        self.d_model = 128
        self.nhead = 8
        self.num_layers = 6
        self.num_classes = 4
        
        # Create ModelConfig object
        self.config = ModelConfig(
            input_dim=self.input_dim,
            sequence_length=self.sequence_length,
            hidden_dim=self.d_model,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            dropout_rate=0.3
        )
        
        self.model = CNNTransformerModel(self.config)
    
    def test_initialization(self):
        """Test transformer model initialization."""
        assert self.model is not None
        assert hasattr(self.model, 'conv1')
        assert hasattr(self.model, 'transformer')
        assert hasattr(self.model, 'pos_encoding')
    
    def test_forward_pass(self):
        """Test forward pass through transformer model."""
        batch_size = 2
        # Fix input shape: models expect (batch, sequence, features)
        input_tensor = torch.randn(batch_size, self.sequence_length, self.input_dim)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        assert output.shape == (batch_size, self.num_classes)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_positional_encoding(self):
        """Test positional encoding."""
        pos_encoding = PositionalEncoding(self.d_model, self.sequence_length)
        
        # Test positional encoding
        x = torch.randn(self.sequence_length, 1, self.d_model)
        encoded = pos_encoding(x)
        
        assert encoded.shape == x.shape
        assert not torch.isnan(encoded).any()


class TestCNNTransformerModelV2:
    """Test cases for CNN-Transformer model V2 with pre-normalization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 20
        self.sequence_length = 60
        self.d_model = 128
        self.nhead = 8
        self.num_layers = 6
        self.num_classes = 4
        
        # Create ModelConfig object
        self.config = ModelConfig(
            input_dim=self.input_dim,
            sequence_length=self.sequence_length,
            hidden_dim=self.d_model,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            dropout_rate=0.3
        )
        
        self.model = CNNTransformerModelV2(self.config)
    
    def test_initialization(self):
        """Test transformer model V2 initialization."""
        assert self.model is not None
        assert hasattr(self.model, 'transformer')
    
    def test_forward_pass(self):
        """Test forward pass through transformer model V2."""
        batch_size = 2
        # Fix input shape: models expect (batch, sequence, features)
        input_tensor = torch.randn(batch_size, self.sequence_length, self.input_dim)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        assert output.shape == (batch_size, self.num_classes)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestModelEnsemble:
    """Test cases for model ensemble."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 20
        self.sequence_length = 60
        self.num_classes = 4
        
        # Create ModelConfig object
        self.config = ModelConfig(
            input_dim=self.input_dim,
            sequence_length=self.sequence_length,
            hidden_dim=64,
            num_layers=2,
            num_classes=self.num_classes,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            dropout_rate=0.3
        )
        
        # Create multiple models
        self.models = [
            CNNLSTMModel(self.config),
            CNNTransformerModel(self.config)
        ]
        
        self.ensemble = ModelEnsemble(self.models)
    
    def test_initialization(self):
        """Test ensemble initialization."""
        assert self.ensemble is not None
        assert len(self.ensemble.models) == 2
    
    def test_weighted_average_prediction(self):
        """Test weighted average ensemble prediction."""
        batch_size = 2
        # Fix input shape: models expect (batch, sequence, features)
        input_tensor = torch.randn(batch_size, self.sequence_length, self.input_dim)
        
        with torch.no_grad():
            prediction = self.ensemble.predict_weighted_average(input_tensor)
        
        assert prediction.shape == (batch_size, self.num_classes)
        assert not torch.isnan(prediction).any()
    
    def test_voting_prediction(self):
        """Test voting ensemble prediction."""
        batch_size = 2
        # Fix input shape: models expect (batch, sequence, features)
        input_tensor = torch.randn(batch_size, self.sequence_length, self.input_dim)
        
        with torch.no_grad():
            prediction = self.ensemble.predict_voting(input_tensor)
        
        assert prediction.shape == (batch_size, self.num_classes)
        assert not torch.isnan(prediction).any()


class TestAdaptiveEnsemble:
    """Test cases for adaptive ensemble."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 20
        self.sequence_length = 60
        self.num_classes = 4
        
        # Create ModelConfig object
        self.config = ModelConfig(
            input_dim=self.input_dim,
            sequence_length=self.sequence_length,
            hidden_dim=64,
            num_layers=2,
            num_classes=self.num_classes,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            dropout_rate=0.3
        )
        
        # Create multiple models
        self.models = [
            CNNLSTMModel(self.config),
            CNNTransformerModel(self.config)
        ]
        
        self.adaptive_ensemble = AdaptiveEnsemble(self.models)
    
    def test_initialization(self):
        """Test adaptive ensemble initialization."""
        assert self.adaptive_ensemble is not None
        assert len(self.adaptive_ensemble.models) == 2
        assert hasattr(self.adaptive_ensemble, 'meta_learner')
    
    def test_adaptive_prediction(self):
        """Test adaptive ensemble prediction."""
        batch_size = 2
        # Fix input shape: models expect (batch, sequence, features)
        input_tensor = torch.randn(batch_size, self.sequence_length, self.input_dim)
        
        with torch.no_grad():
            prediction = self.adaptive_ensemble.predict(input_tensor)
        
        # predict() returns class indices, not logits
        assert prediction.shape == (batch_size,)
        assert not torch.isnan(prediction).any()


class TestModelTrainer:
    """Test cases for model trainer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model_config = ModelConfig(
            input_dim=20,
            sequence_length=60,
            hidden_dim=64,
            num_layers=2,
            num_classes=4,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            dropout_rate=0.3
        )
        
        self.model = CNNLSTMModel(self.model_config)
        
        self.trainer = ModelTrainer(self.model, self.model_config)
    
    def test_initialization(self):
        """Test trainer initialization."""
        assert self.trainer is not None
        assert self.trainer.model is not None
        assert self.trainer.config is not None
    
    def test_setup_training(self):
        """Test training setup."""
        self.trainer.setup_training(
            learning_rate=0.001,
            weight_decay=1e-5
        )
        
        assert self.trainer.optimizer is not None
        assert self.trainer.scheduler is not None
        assert self.trainer.criterion is not None
    
    def test_create_data_loaders(self):
        """Test data loader creation."""
        # Create sample data
        X = np.random.randn(100, 60, 20)  # 100 samples, 60 timesteps, 20 features
        y = np.random.randint(0, 4, 100)  # 100 labels
        
        train_loader, val_loader, test_loader = create_data_loaders(
            X, y, self.model_config
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Test data loader iteration
        for batch_x, batch_y in train_loader:
            assert batch_x.shape[0] <= 32  # batch size
            assert batch_x.shape[1] == 60  # sequence length
            assert batch_x.shape[2] == 20  # input dim
            assert batch_y.shape[0] == batch_x.shape[0]
            break  # Just test first batch
    
    def test_train_epoch(self):
        """Test training epoch."""
        # Setup training
        self.trainer.setup_training()
        
        # Create sample data
        X = torch.randn(32, 60, 20)  # batch_size, sequence_length, input_dim
        y = torch.randint(0, 4, (32,))  # batch_size labels
        
        # Create data loader
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Test training epoch
        train_loss, train_acc = self.trainer.train_epoch(dataloader)
        
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert train_loss >= 0
        assert 0 <= train_acc <= 100
    
    def test_validate(self):
        """Test validation."""
        # Setup training
        self.trainer.setup_training()
        
        # Create sample data
        X = torch.randn(32, 60, 20)  # batch_size, sequence_length, input_dim
        y = torch.randint(0, 4, (32,))
        
        # Create data loader
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Test validation
        val_loss, val_acc = self.trainer.validate(dataloader)
        
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert val_loss >= 0
        assert 0 <= val_acc <= 100


class TestModelIntegration:
    """Integration tests for model components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 20
        self.sequence_length = 60
        self.num_classes = 4
        
        # Create sample data
        self.X = np.random.randn(100, 60, 20)
        self.y = np.random.randint(0, 4, 100)
    
    def test_model_comparison(self):
        """Test comparison between different models."""
        # Create ModelConfig object
        config = ModelConfig(
            input_dim=self.input_dim,
            sequence_length=self.sequence_length,
            hidden_dim=64,
            num_layers=2,
            num_classes=self.num_classes,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            dropout_rate=0.3
        )
        
        # CNN-LSTM model
        cnn_lstm = CNNLSTMModel(config)
        
        # CNN-Transformer model
        cnn_transformer = CNNTransformerModel(config)
        
        # Test both models with same input
        # Fix input shape: models expect (batch, sequence, features)
        input_tensor = torch.randn(2, self.sequence_length, self.input_dim)
        
        with torch.no_grad():
            lstm_output = cnn_lstm(input_tensor)
            transformer_output = cnn_transformer(input_tensor)
        
        assert lstm_output.shape == transformer_output.shape
        assert lstm_output.shape == (2, self.num_classes)
    
    def test_ensemble_creation(self):
        """Test ensemble model creation."""
        # Create ModelConfig object
        config = ModelConfig(
            input_dim=self.input_dim,
            sequence_length=self.sequence_length,
            hidden_dim=64,
            num_layers=2,
            num_classes=self.num_classes,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            dropout_rate=0.3
        )
        
        models = [
            CNNLSTMModel(config),
            CNNTransformerModel(config)
        ]
        
        ensemble = ModelEnsemble(models, ensemble_method='weighted_average')
        
        assert ensemble is not None
        assert hasattr(ensemble, 'predict')
        
        # Test ensemble prediction
        # Fix input shape: models expect (batch, sequence, features)
        input_tensor = torch.randn(2, self.sequence_length, self.input_dim)
        
        with torch.no_grad():
            prediction = ensemble.predict(input_tensor)
        
        # predict() returns class indices, not logits
        assert prediction.shape == (2,)
        assert not torch.isnan(prediction).any()


if __name__ == "__main__":
    pytest.main([__file__])
