"""
Unit tests for the Differentiable Mixed Precision Search Algorithm.

Tests cover:
1. SearchConfig dataclass and validation
2. DifferentiableMixedPrecisionSearch initialization
3. Loss computation (energy, bit, total)
4. Search step execution
5. Full search process
6. AutoBitSuperNet functionality
7. Integration with search space
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from co_nas.search_space import (
    CoNASSearchSpace,
    ArchitectureConfig,
    LayerConfig,
    PrecisionLevel,
)
from co_nas.search_algorithm import (
    SearchConfig,
    DifferentiableMixedPrecisionSearch,
    AutoBitSuperNet,
)


class TestSearchConfig:
    """Tests for SearchConfig dataclass."""
    
    def test_default_configuration(self):
        """Test default search configuration values."""
        config = SearchConfig()
        
        assert config.arch_lr == 0.01
        assert config.precision_lr == 0.01
        assert config.weight_decay == 3e-4
        assert config.temperature == 1.0
        assert config.temperature_decay == 0.995
        assert config.min_temperature == 0.1
        assert config.energy_weight == 0.1
        assert config.bit_weight == 0.05
        assert config.num_search_steps == 1000
        assert config.validation_frequency == 50
        assert config.energy_budget is None
        assert config.bit_budget is None
    
    def test_custom_configuration(self):
        """Test custom search configuration."""
        config = SearchConfig(
            arch_lr=0.001,
            precision_lr=0.001,
            energy_budget=1000.0,
            bit_budget=1000000,
            temperature=0.5,
            energy_weight=0.2,
            bit_weight=0.1,
            num_search_steps=500,
        )
        
        assert config.arch_lr == 0.001
        assert config.precision_lr == 0.001
        assert config.energy_budget == 1000.0
        assert config.bit_budget == 1000000
        assert config.temperature == 0.5
        assert config.energy_weight == 0.2
        assert config.bit_weight == 0.1
        assert config.num_search_steps == 500
    
    def test_energy_weight_validation_positive(self):
        """Test that energy_weight must be in [0, 1]."""
        # Valid values
        config1 = SearchConfig(energy_weight=0.0)
        assert config1.energy_weight == 0.0
        
        config2 = SearchConfig(energy_weight=1.0)
        assert config2.energy_weight == 1.0
        
        config3 = SearchConfig(energy_weight=0.5)
        assert config3.energy_weight == 0.5
    
    def test_energy_weight_validation_negative(self):
        """Test that negative energy_weight raises error."""
        with pytest.raises(ValueError, match="energy_weight must be in"):
            SearchConfig(energy_weight=-0.1)
    
    def test_energy_weight_validation_too_large(self):
        """Test that energy_weight > 1 raises error."""
        with pytest.raises(ValueError, match="energy_weight must be in"):
            SearchConfig(energy_weight=1.1)
    
    def test_bit_weight_validation(self):
        """Test bit_weight validation."""
        # Valid
        config = SearchConfig(bit_weight=0.5)
        assert config.bit_weight == 0.5
        
        # Invalid - negative
        with pytest.raises(ValueError, match="bit_weight must be in"):
            SearchConfig(bit_weight=-0.1)
        
        # Invalid - too large
        with pytest.raises(ValueError, match="bit_weight must be in"):
            SearchConfig(bit_weight=1.1)


class TestDifferentiableMixedPrecisionSearch:
    """Tests for DifferentiableMixedPrecisionSearch class."""
    
    @pytest.fixture
    def search_space(self):
        """Create a small search space for testing."""
        return CoNASSearchSpace(num_layers=3, base_channels=16)
    
    @pytest.fixture
    def search_config(self):
        """Create a minimal search config for testing."""
        return SearchConfig(
            num_search_steps=5,
            validation_frequency=2,
            min_temperature=0.1,
        )
    
    @pytest.fixture
    def search_algorithm(self, search_space, search_config):
        """Create search algorithm instance."""
        return DifferentiableMixedPrecisionSearch(
            search_space=search_space,
            search_config=search_config,
            device="cpu",
        )
    
    def test_initialization(self, search_space, search_config):
        """Test search algorithm initialization."""
        algorithm = DifferentiableMixedPrecisionSearch(
            search_space=search_space,
            search_config=search_config,
            device="cpu",
        )
        
        assert algorithm.search_space is not None
        assert algorithm.config == search_config
        assert algorithm.device == "cpu"
        assert algorithm.arch_optimizer is None
        assert algorithm.precision_optimizer is None
        assert algorithm.best_architecture is None
        assert algorithm.best_score == float("-inf")
    
    def test_initialization_default_config(self, search_space):
        """Test initialization with default config."""
        algorithm = DifferentiableMixedPrecisionSearch(
            search_space=search_space,
            device="cpu",
        )
        
        assert algorithm.config is not None
        assert isinstance(algorithm.config, SearchConfig)
    
    def test_compute_energy_loss(self, search_algorithm):
        """Test energy loss computation."""
        # Create a simple architecture
        layers = [
            LayerConfig("conv", 16, 32, PrecisionLevel.INT8, kernel_size=3),
            LayerConfig("conv", 32, 64, PrecisionLevel.INT4, kernel_size=3),
        ]
        arch_config = ArchitectureConfig(layers=layers, skip_connections=[])
        
        energy_loss = search_algorithm.compute_energy_loss(arch_config)
        
        assert isinstance(energy_loss, torch.Tensor)
        assert energy_loss.item() > 0
        assert energy_loss.device.type == "cpu"
    
    def test_compute_energy_loss_with_budget(self, search_space):
        """Test energy loss computation with budget normalization."""
        config = SearchConfig(energy_budget=1000.0)
        algorithm = DifferentiableMixedPrecisionSearch(
            search_space=search_space,
            search_config=config,
            device="cpu",
        )
        
        layers = [
            LayerConfig("conv", 16, 32, PrecisionLevel.INT8, kernel_size=3),
        ]
        arch_config = ArchitectureConfig(layers=layers, skip_connections=[])
        
        energy_loss = algorithm.compute_energy_loss(arch_config)
        
        # With budget normalization, loss should be relative to budget
        assert isinstance(energy_loss, torch.Tensor)
        assert energy_loss.item() > 0
    
    def test_compute_bit_loss(self, search_algorithm):
        """Test bit loss computation."""
        precision_weights = search_algorithm.search_space.get_precision_weights()
        
        bit_loss = search_algorithm.compute_bit_loss(precision_weights)
        
        assert isinstance(bit_loss, torch.Tensor)
        assert bit_loss.item() >= 0
        assert bit_loss.item() <= 2.0  # Should be normalized
    
    def test_compute_bit_loss_with_budget(self, search_space):
        """Test bit loss computation with budget constraint."""
        config = SearchConfig(bit_budget=10000)
        algorithm = DifferentiableMixedPrecisionSearch(
            search_space=search_space,
            search_config=config,
            device="cpu",
        )
        
        precision_weights = algorithm.search_space.get_precision_weights()
        bit_loss = algorithm.compute_bit_loss(precision_weights)
        
        assert isinstance(bit_loss, torch.Tensor)
        assert bit_loss.item() >= 0
    
    def test_compute_total_loss(self, search_algorithm):
        """Test total loss computation."""
        val_loss = torch.tensor(1.0)
        energy_loss = torch.tensor(0.5)
        bit_loss = torch.tensor(0.3)
        
        total_loss = search_algorithm.compute_total_loss(
            val_loss, energy_loss, bit_loss
        )
        
        expected = 1.0 + 0.1 * 0.5 + 0.05 * 0.3
        assert abs(total_loss.item() - expected) < 1e-5
    
    def test_compute_total_loss_with_custom_weights(self, search_space):
        """Test total loss with custom regularization weights."""
        config = SearchConfig(energy_weight=0.2, bit_weight=0.1)
        algorithm = DifferentiableMixedPrecisionSearch(
            search_space=search_space,
            search_config=config,
            device="cpu",
        )
        
        val_loss = torch.tensor(1.0)
        energy_loss = torch.tensor(0.5)
        bit_loss = torch.tensor(0.3)
        
        total_loss = algorithm.compute_total_loss(val_loss, energy_loss, bit_loss)
        
        expected = 1.0 + 0.2 * 0.5 + 0.1 * 0.3
        assert abs(total_loss.item() - expected) < 1e-5
    
    def test_initialize_optimizers(self, search_algorithm):
        """Test optimizer initialization."""
        assert search_algorithm.arch_optimizer is None
        assert search_algorithm.precision_optimizer is None
        
        search_algorithm._initialize_optimizers()
        
        assert search_algorithm.arch_optimizer is not None
        assert search_algorithm.precision_optimizer is not None
        
        # Check optimizer types
        assert isinstance(search_algorithm.arch_optimizer, torch.optim.Adam)
        assert isinstance(search_algorithm.precision_optimizer, torch.optim.Adam)
    
    def test_search_step(self, search_algorithm):
        """Test single search step."""
        # Create dummy data
        train_data = torch.randn(4, 3, 32, 32)
        train_targets = torch.randint(0, 10, (4,))
        val_data = torch.randn(4, 3, 32, 32)
        val_targets = torch.randint(0, 10, (4,))
        
        # Create simple model that accepts arch_config
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.flatten = nn.Flatten()
                self.linear = nn.Linear(16, 10)
            
            def forward(self, x, arch_config=None):
                out = self.conv(x)
                out = torch.relu(out)
                out = self.pool(out)
                out = self.flatten(out)
                return self.linear(out)
        
        model = SimpleModel()
        
        metrics = search_algorithm.search_step(
            model=model,
            train_data=train_data,
            train_targets=train_targets,
            val_data=val_data,
            val_targets=val_targets,
        )
        
        # Check metrics returned
        assert "total_loss" in metrics
        assert "validation_loss" in metrics
        assert "energy_loss" in metrics
        assert "bit_loss" in metrics
        assert "temperature" in metrics
        
        # Check values are valid
        assert isinstance(metrics["total_loss"], float)
        assert metrics["total_loss"] > 0
        
        # Check temperature decayed
        assert metrics["temperature"] <= search_algorithm.config.temperature * 1.01
    
    def test_search_step_creates_optimizers(self, search_space):
        """Test that search_step initializes optimizers if needed."""
        algorithm = DifferentiableMixedPrecisionSearch(
            search_space=search_space,
            device="cpu",
        )
        
        assert algorithm.arch_optimizer is None
        
        # Run search step
        train_data = torch.randn(2, 3, 32, 32)
        train_targets = torch.randint(0, 10, (2,))
        val_data = torch.randn(2, 3, 32, 32)
        val_targets = torch.randint(0, 10, (2,))
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
            
            def forward(self, x, arch_config=None):
                return self.conv(x).mean(dim=(2, 3))
        
        model = SimpleModel()
        
        algorithm.search_step(model, train_data, train_targets, val_data, val_targets)
        
        # Optimizers should now be initialized
        assert algorithm.arch_optimizer is not None
        assert algorithm.precision_optimizer is not None
    
    def test_get_search_statistics_empty(self, search_space):
        """Test statistics retrieval when no search has been run."""
        algorithm = DifferentiableMixedPrecisionSearch(
            search_space=search_space,
            device="cpu",
        )
        
        stats = algorithm.get_search_statistics()
        assert stats == {}
    
    def test_search_integration(self, search_space):
        """Test full search process integration."""
        config = SearchConfig(
            num_search_steps=3,
            validation_frequency=1,
        )
        algorithm = DifferentiableMixedPrecisionSearch(
            search_space=search_space,
            search_config=config,
            device="cpu",
        )
        
        # Create simple model that accepts arch_config (not full SuperNet to save memory)
        class SimpleSuperNet(nn.Module):
            def __init__(self, base_channels=16, num_classes=10):
                super().__init__()
                self.conv = nn.Conv2d(3, base_channels, 3, padding=1)
                self.classifier = nn.Linear(base_channels, num_classes)
            
            def forward(self, x, arch_config=None):
                out = self.conv(x)
                out = torch.relu(out)
                out = torch.mean(out, dim=(2, 3))
                return self.classifier(out)
        
        super_net = SimpleSuperNet(base_channels=16, num_classes=10)
        
        # Create dummy data loaders
        train_data = torch.randn(8, 3, 32, 32)
        train_targets = torch.randint(0, 10, (8,))
        val_data = torch.randn(8, 3, 32, 32)
        val_targets = torch.randint(0, 10, (8,))
        
        train_dataset = TensorDataset(train_data, train_targets)
        val_dataset = TensorDataset(val_data, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=4)
        val_loader = DataLoader(val_dataset, batch_size=4)
        
        # Run search
        best_arch = algorithm.search(super_net, train_loader, val_loader)
        
        # Check results
        assert best_arch is not None
        assert isinstance(best_arch, ArchitectureConfig)
        assert len(best_arch.layers) == search_space.num_layers
        
        # Check statistics recorded
        stats = algorithm.get_search_statistics()
        assert stats["num_steps"] == 3
        assert "final_loss" in stats
        assert "best_score" in stats


class TestAutoBitSuperNet:
    """Tests for AutoBitSuperNet class."""
    
    def test_initialization_default(self):
        """Test SuperNet initialization with small default parameters to save memory."""
        net = AutoBitSuperNet(base_channels=8, num_classes=10, input_size=16)
        
        assert net.base_channels == 8
        assert net.num_classes == 10
        assert net.input_size == 16
        assert len(net.conv_layers) > 0
        assert len(net.linear_layers) > 0
    
    def test_initialization_custom(self):
        """Test SuperNet initialization with custom parameters."""
        net = AutoBitSuperNet(
            base_channels=8,
            num_classes=100,
            input_size=16,
        )
        
        assert net.base_channels == 8
        assert net.num_classes == 100
        assert net.input_size == 16
    
    def test_forward_pass(self):
        """Test forward pass through SuperNet."""
        net = AutoBitSuperNet(base_channels=16, num_classes=10, input_size=32)
        
        # Create architecture config
        layers = [
            LayerConfig("conv", 16, 32, PrecisionLevel.INT8, kernel_size=3),
            LayerConfig("conv", 32, 64, PrecisionLevel.INT8, kernel_size=3),
        ]
        arch_config = ArchitectureConfig(layers=layers, skip_connections=[])
        
        # Forward pass
        x = torch.randn(2, 3, 32, 32)
        output = net(x, arch_config)
        
        # Check output shape
        assert output.shape[0] == 2
        assert output.shape[1] == 10  # num_classes
    
    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        net = AutoBitSuperNet(base_channels=16, num_classes=10, input_size=32)
        
        layers = [
            LayerConfig("conv", 16, 32, PrecisionLevel.INT8, kernel_size=3),
        ]
        arch_config = ArchitectureConfig(layers=layers, skip_connections=[])
        
        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 3, 32, 32)
            output = net(x, arch_config)
            assert output.shape[0] == batch_size
    
    def test_parameters_exist(self):
        """Test that SuperNet has learnable parameters."""
        net = AutoBitSuperNet(base_channels=16)
        
        params = list(net.parameters())
        assert len(params) > 0
        
        # All parameters should require gradients
        for param in params:
            assert param.requires_grad
    
    def test_device_compatibility_cpu(self):
        """Test SuperNet on CPU."""
        net = AutoBitSuperNet(base_channels=16)
        
        # Should work on CPU by default
        assert any(p.device.type == "cpu" for p in net.parameters())
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compatibility_cuda(self):
        """Test SuperNet on CUDA."""
        net = AutoBitSuperNet(base_channels=16).cuda()
        
        assert any(p.device.type == "cuda" for p in net.parameters())
    
    def test_gradient_flow(self):
        """Test that gradients flow through SuperNet."""
        net = AutoBitSuperNet(base_channels=16, num_classes=10, input_size=32)
        
        layers = [
            LayerConfig("conv", 16, 32, PrecisionLevel.INT8, kernel_size=3),
        ]
        arch_config = ArchitectureConfig(layers=layers, skip_connections=[])
        
        x = torch.randn(2, 3, 32, 32)
        output = net(x, arch_config)
        
        # Compute loss and backpropagate
        loss = output.sum()
        loss.backward()
        
        # Check that at least some parameters have gradients
        # (In a SuperNet, only active paths receive gradients)
        params_with_grad = [p for p in net.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0
        
        # The first conv layer and classifier should have gradients
        assert net.conv_layers[0].weight.grad is not None
        assert net.classifier.weight.grad is not None


class TestIntegration:
    """Integration tests for the complete search pipeline."""
    
    def test_end_to_end_search(self):
        """Test complete end-to-end search process."""
        # Create search space
        search_space = CoNASSearchSpace(num_layers=3, base_channels=16)
        
        # Create search config
        config = SearchConfig(
            num_search_steps=5,
            validation_frequency=2,
            energy_weight=0.1,
            bit_weight=0.05,
        )
        
        # Create search algorithm
        algorithm = DifferentiableMixedPrecisionSearch(
            search_space=search_space,
            search_config=config,
            device="cpu",
        )
        
        # Create SuperNet
        super_net = AutoBitSuperNet(base_channels=16, num_classes=10, input_size=32)
        
        # Create data
        train_data = torch.randn(16, 3, 32, 32)
        train_targets = torch.randint(0, 10, (16,))
        val_data = torch.randn(16, 3, 32, 32)
        val_targets = torch.randint(0, 10, (16,))
        
        train_dataset = TensorDataset(train_data, train_targets)
        val_dataset = TensorDataset(val_data, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=4)
        val_loader = DataLoader(val_dataset, batch_size=4)
        
        # Callback to track progress
        callback_called = []
        
        def callback(step, metrics):
            callback_called.append((step, metrics))
        
        # Run search
        best_arch = algorithm.search(
            super_net, train_loader, val_loader, callback=callback
        )
        
        # Verify results
        assert best_arch is not None
        assert isinstance(best_arch, ArchitectureConfig)
        assert len(best_arch.layers) == 3
        
        # Verify callback was called
        assert len(callback_called) > 0
        
        # Verify statistics
        stats = algorithm.get_search_statistics()
        assert stats["num_steps"] == 5
        assert "final_loss" in stats
        assert "best_score" in stats
    
    def test_search_with_energy_budget(self):
        """Test search with energy budget constraint."""
        search_space = CoNASSearchSpace(num_layers=3, base_channels=16)
        
        config = SearchConfig(
            num_search_steps=3,
            energy_budget=1000.0,
            energy_weight=0.2,
        )
        
        algorithm = DifferentiableMixedPrecisionSearch(
            search_space=search_space,
            search_config=config,
            device="cpu",
        )
        
        super_net = AutoBitSuperNet(base_channels=16, num_classes=10, input_size=32)
        
        train_data = torch.randn(8, 3, 32, 32)
        train_targets = torch.randint(0, 10, (8,))
        val_data = torch.randn(8, 3, 32, 32)
        val_targets = torch.randint(0, 10, (8,))
        
        train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=4)
        val_loader = DataLoader(TensorDataset(val_data, val_targets), batch_size=4)
        
        best_arch = algorithm.search(super_net, train_loader, val_loader)
        
        assert best_arch is not None
        
        # Check energy is within reasonable range
        energy = search_space.estimate_energy(best_arch)
        assert energy > 0
    
    def test_search_with_bit_budget(self):
        """Test search with bit budget constraint."""
        search_space = CoNASSearchSpace(num_layers=3, base_channels=16)
        
        config = SearchConfig(
            num_search_steps=3,
            bit_budget=100000,
            bit_weight=0.1,
        )
        
        algorithm = DifferentiableMixedPrecisionSearch(
            search_space=search_space,
            search_config=config,
            device="cpu",
        )
        
        super_net = AutoBitSuperNet(base_channels=16, num_classes=10, input_size=32)
        
        train_data = torch.randn(8, 3, 32, 32)
        train_targets = torch.randint(0, 10, (8,))
        
        train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=4)
        val_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=4)
        
        best_arch = algorithm.search(super_net, train_loader, val_loader)
        
        assert best_arch is not None
        
        # Check total bits
        total_bits = best_arch.total_bits()
        assert total_bits > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
