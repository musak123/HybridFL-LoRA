"""
Unit tests for the Co-NAS Energy-Aware Joint Search Space.

Tests cover:
1. PrecisionLevel enum functionality
2. LayerConfig dataclass
3. ArchitectureConfig dataclass
4. EnergyEstimator calculations
5. CoNASSearchSpace initialization and methods
6. Differentiable parameter handling
7. Architecture sampling
8. Energy estimation accuracy
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from co_nas.search_space import (
    PrecisionLevel,
    LayerConfig,
    ArchitectureConfig,
    EnergyEstimator,
    CoNASSearchSpace,
)


class TestPrecisionLevel:
    """Tests for PrecisionLevel enum."""
    
    def test_precision_values(self):
        """Test that precision levels have correct bit values."""
        assert PrecisionLevel.FP32.value == 32
        assert PrecisionLevel.FP16.value == 16
        assert PrecisionLevel.INT8.value == 8
        assert PrecisionLevel.INT4.value == 4
        assert PrecisionLevel.INT2.value == 2
    
    def test_precision_ordering(self):
        """Test that higher precision has larger values."""
        assert PrecisionLevel.FP32.value > PrecisionLevel.FP16.value
        assert PrecisionLevel.FP16.value > PrecisionLevel.INT8.value
        assert PrecisionLevel.INT8.value > PrecisionLevel.INT4.value
        assert PrecisionLevel.INT4.value > PrecisionLevel.INT2.value
    
    def test_all_precisions_present(self):
        """Test that all expected precision levels are available."""
        precisions = [p.value for p in PrecisionLevel]
        assert 32 in precisions
        assert 16 in precisions
        assert 8 in precisions
        assert 4 in precisions
        assert 2 in precisions


class TestLayerConfig:
    """Tests for LayerConfig dataclass."""
    
    def test_layer_config_creation_conv(self):
        """Test creating a convolutional layer config."""
        config = LayerConfig(
            layer_type="conv",
            in_features=64,
            out_features=128,
            precision=PrecisionLevel.INT8,
            kernel_size=3,
            stride=1,
        )
        
        assert config.layer_type == "conv"
        assert config.in_features == 64
        assert config.out_features == 128
        assert config.precision == PrecisionLevel.INT8
        assert config.kernel_size == 3
        assert config.stride == 1
        assert config.activation == "relu"  # default
    
    def test_layer_config_creation_linear(self):
        """Test creating a linear layer config."""
        config = LayerConfig(
            layer_type="linear",
            in_features=256,
            out_features=10,
            precision=PrecisionLevel.FP16,
        )
        
        assert config.layer_type == "linear"
        assert config.kernel_size is None
        assert config.stride is None
    
    def test_get_bit_width(self):
        """Test bit width retrieval for different precisions."""
        config_fp32 = LayerConfig(
            layer_type="conv",
            in_features=64,
            out_features=64,
            precision=PrecisionLevel.FP32,
        )
        assert config_fp32.get_bit_width() == 32
        
        config_int8 = LayerConfig(
            layer_type="conv",
            in_features=64,
            out_features=64,
            precision=PrecisionLevel.INT8,
        )
        assert config_int8.get_bit_width() == 8
        
        config_int4 = LayerConfig(
            layer_type="conv",
            in_features=64,
            out_features=64,
            precision=PrecisionLevel.INT4,
        )
        assert config_int4.get_bit_width() == 4


class TestArchitectureConfig:
    """Tests for ArchitectureConfig dataclass."""
    
    def test_architecture_config_creation(self):
        """Test creating an architecture config."""
        layers = [
            LayerConfig("conv", 64, 128, PrecisionLevel.INT8, kernel_size=3),
            LayerConfig("conv", 128, 256, PrecisionLevel.INT8, kernel_size=3),
            LayerConfig("linear", 256, 10, PrecisionLevel.FP16),
        ]
        skip_connections = [(0, 2)]
        
        config = ArchitectureConfig(layers=layers, skip_connections=skip_connections)
        
        assert len(config.layers) == 3
        assert len(config.skip_connections) == 1
        assert config.skip_connections[0] == (0, 2)
    
    def test_num_layers(self):
        """Test layer counting."""
        layers = [
            LayerConfig("conv", 64, 128, PrecisionLevel.INT8, kernel_size=3),
            LayerConfig("conv", 128, 256, PrecisionLevel.INT8, kernel_size=3),
        ]
        config = ArchitectureConfig(layers=layers, skip_connections=[])
        assert config.num_layers() == 2
    
    def test_total_bits_conv_layer(self):
        """Test total bits calculation for convolutional layers."""
        # Conv layer: 3x3 kernel, 64 input channels, 128 output channels
        # Parameters: 3 * 3 * 64 * 128 = 73,728
        # With INT8 precision: 73,728 * 8 = 589,824 bits
        layers = [
            LayerConfig("conv", 64, 128, PrecisionLevel.INT8, kernel_size=3),
        ]
        config = ArchitectureConfig(layers=layers, skip_connections=[])
        
        expected_bits = 3 * 3 * 64 * 128 * 8
        assert config.total_bits() == expected_bits
    
    def test_total_bits_linear_layer(self):
        """Test total bits calculation for linear layers."""
        # Linear layer: 256 input, 10 output
        # Parameters: 256 * 10 = 2,560
        # With FP16 precision: 2,560 * 16 = 40,960 bits
        layers = [
            LayerConfig("linear", 256, 10, PrecisionLevel.FP16),
        ]
        config = ArchitectureConfig(layers=layers, skip_connections=[])
        
        expected_bits = 256 * 10 * 16
        assert config.total_bits() == expected_bits
    
    def test_total_bits_multiple_layers(self):
        """Test total bits calculation for multiple layers."""
        layers = [
            LayerConfig("conv", 64, 128, PrecisionLevel.INT8, kernel_size=3),
            LayerConfig("conv", 128, 256, PrecisionLevel.INT4, kernel_size=3),
        ]
        config = ArchitectureConfig(layers=layers, skip_connections=[])
        
        # Layer 1: 3*3*64*128*8 = 589,824
        # Layer 2: 3*3*128*256*4 = 1,179,648
        expected_bits = 589824 + 1179648
        assert config.total_bits() == expected_bits


class TestEnergyEstimator:
    """Tests for EnergyEstimator class."""
    
    def test_energy_costs_defined(self):
        """Test that energy costs are defined for all precisions."""
        for precision in PrecisionLevel:
            assert precision in EnergyEstimator.ENERGY_COSTS
            assert "mac" in EnergyEstimator.ENERGY_COSTS[precision]
            assert "memory" in EnergyEstimator.ENERGY_COSTS[precision]
    
    def test_energy_cost_ordering(self):
        """Test that lower precision has lower energy costs."""
        costs = EnergyEstimator.ENERGY_COSTS
        
        # MAC operations should be cheaper for lower precision
        assert costs[PrecisionLevel.FP32]["mac"] > costs[PrecisionLevel.FP16]["mac"]
        assert costs[PrecisionLevel.FP16]["mac"] > costs[PrecisionLevel.INT8]["mac"]
        assert costs[PrecisionLevel.INT8]["mac"] > costs[PrecisionLevel.INT4]["mac"]
        assert costs[PrecisionLevel.INT4]["mac"] > costs[PrecisionLevel.INT2]["mac"]
    
    def test_estimate_layer_energy_conv(self):
        """Test energy estimation for convolutional layer."""
        config = LayerConfig(
            layer_type="conv",
            in_features=64,
            out_features=128,
            precision=PrecisionLevel.INT8,
            kernel_size=3,
        )
        
        # Number of operations: 3*3*64*128 = 73,728
        num_ops = 3 * 3 * 64 * 128
        
        energy = EnergyEstimator.estimate_layer_energy(config, num_ops)
        
        # INT8: mac=0.4, memory=0.1, total=0.5 pJ per op
        expected_energy = num_ops * (0.4 + 0.1)
        assert abs(energy - expected_energy) < 1e-5
    
    def test_estimate_layer_energy_different_precisions(self):
        """Test energy estimation varies with precision."""
        num_ops = 10000
        
        energy_fp32 = EnergyEstimator.estimate_layer_energy(
            LayerConfig("conv", 64, 64, PrecisionLevel.FP32, kernel_size=3),
            num_ops
        )
        energy_int8 = EnergyEstimator.estimate_layer_energy(
            LayerConfig("conv", 64, 64, PrecisionLevel.INT8, kernel_size=3),
            num_ops
        )
        energy_int4 = EnergyEstimator.estimate_layer_energy(
            LayerConfig("conv", 64, 64, PrecisionLevel.INT4, kernel_size=3),
            num_ops
        )
        
        assert energy_fp32 > energy_int8
        assert energy_int8 > energy_int4
    
    def test_estimate_architecture_energy(self):
        """Test energy estimation for complete architecture."""
        layers = [
            LayerConfig("conv", 64, 128, PrecisionLevel.INT8, kernel_size=3),
            LayerConfig("conv", 128, 256, PrecisionLevel.INT4, kernel_size=3),
            LayerConfig("linear", 256, 10, PrecisionLevel.FP16),
        ]
        config = ArchitectureConfig(layers=layers, skip_connections=[])
        
        energy = EnergyEstimator.estimate_architecture_energy(config)
        
        # Energy should be positive
        assert energy > 0
        
        # Calculate expected energy manually
        expected_energy = 0.0
        
        # Layer 1: conv INT8
        ops1 = 3 * 3 * 64 * 128
        expected_energy += ops1 * (0.4 + 0.1)
        
        # Layer 2: conv INT4
        ops2 = 3 * 3 * 128 * 256
        expected_energy += ops2 * (0.2 + 0.05)
        
        # Layer 3: linear FP16
        ops3 = 256 * 10
        expected_energy += ops3 * (1.1 + 0.2)
        
        assert abs(energy - expected_energy) < 1e-5


class TestCoNASSearchSpace:
    """Tests for CoNASSearchSpace class."""
    
    def test_initialization_default(self):
        """Test search space initialization with default parameters."""
        search_space = CoNASSearchSpace()
        
        assert search_space.num_layers == 10
        assert search_space.base_channels == 64
        assert search_space.max_skip_connections == 5
        assert len(search_space.precision_levels) == 4
    
    def test_initialization_custom(self):
        """Test search space initialization with custom parameters."""
        custom_precisions = [PrecisionLevel.FP32, PrecisionLevel.INT8]
        search_space = CoNASSearchSpace(
            num_layers=5,
            base_channels=32,
            precision_levels=custom_precisions,
            max_skip_connections=3,
        )
        
        assert search_space.num_layers == 5
        assert search_space.base_channels == 32
        assert len(search_space.precision_levels) == 2
        assert search_space.max_skip_connections == 3
    
    def test_arch_params_initialized(self):
        """Test that architecture parameters are properly initialized."""
        search_space = CoNASSearchSpace(num_layers=5)
        
        # Check layer type parameters
        for i in range(5):
            assert f"layer_type_{i}" in search_space.arch_params
            assert search_space.arch_params[f"layer_type_{i}"].shape[0] == 4  # 4 layer types
            
            assert f"channels_{i}" in search_space.arch_params
            assert search_space.arch_params[f"channels_{i}"].shape[0] == 4  # 4 channel ratios
        
        # Check skip connection parameters
        assert "skip_connections" in search_space.arch_params
        assert search_space.arch_params["skip_connections"].shape == (5, 5)
    
    def test_precision_params_initialized(self):
        """Test that precision parameters are properly initialized."""
        search_space = CoNASSearchSpace(num_layers=5)
        
        for i in range(5):
            assert f"layer_{i}" in search_space.precision_params
            assert len(search_space.precision_params[f"layer_{i}"]) == len(search_space.precision_levels)
    
    def test_parameters_are_differentiable(self):
        """Test that parameters require gradients."""
        search_space = CoNASSearchSpace(num_layers=3)
        
        for param in search_space.arch_params.values():
            assert param.requires_grad
        
        for param in search_space.precision_params.values():
            assert param.requires_grad
    
    def test_get_architecture_weights(self):
        """Test architecture weight computation."""
        search_space = CoNASSearchSpace(num_layers=3)
        weights = search_space.get_architecture_weights()
        
        # Check all expected keys exist
        for i in range(3):
            assert f"layer_type_{i}" in weights
            assert f"channels_{i}" in weights
        
        assert "skip_connections" in weights
        
        # Check shapes
        for i in range(3):
            assert weights[f"layer_type_{i}"].shape[0] == 4
            assert weights[f"channels_{i}"].shape[0] == 4
        
        assert weights["skip_connections"].shape == (3, 3)
    
    def test_architecture_weights_sum_to_one(self):
        """Test that softmax-normalized weights sum to one."""
        search_space = CoNASSearchSpace(num_layers=3)
        weights = search_space.get_architecture_weights()
        
        for i in range(3):
            # Layer type probabilities should sum to 1
            assert abs(weights[f"layer_type_{i}"].sum().item() - 1.0) < 1e-5
            
            # Channel ratio probabilities should sum to 1
            assert abs(weights[f"channels_{i}"].sum().item() - 1.0) < 1e-5
    
    def test_get_precision_weights(self):
        """Test precision weight computation."""
        search_space = CoNASSearchSpace(num_layers=3)
        weights = search_space.get_precision_weights()
        
        assert len(weights) == 3
        
        for i in range(3):
            assert i in weights
            # Weights should sum to 1 (softmax)
            assert abs(weights[i].sum().item() - 1.0) < 1e-5
    
    def test_sample_architecture(self):
        """Test architecture sampling."""
        search_space = CoNASSearchSpace(num_layers=5)
        arch_config = search_space.sample_architecture(temperature=1.0)
        
        assert isinstance(arch_config, ArchitectureConfig)
        assert len(arch_config.layers) == 5
        assert len(arch_config.skip_connections) <= search_space.max_skip_connections
        
        # Check all layers have valid configurations
        for layer in arch_config.layers:
            assert layer.layer_type in ["conv", "linear", "pool", "identity"]
            assert layer.precision in search_space.precision_levels
    
    def test_sample_architecture_temperature(self):
        """Test that temperature affects sampling determinism."""
        search_space = CoNASSearchSpace(num_layers=5)
        
        # Low temperature should give more deterministic results
        arch_low_temp = search_space.sample_architecture(temperature=0.1)
        
        # Very low temperature should always pick argmax
        arch_very_low_temp = search_space.sample_architecture(temperature=0.01)
        
        # Both should have same number of layers
        assert len(arch_low_temp.layers) == len(arch_very_low_temp.layers) == 5
    
    def test_estimate_energy(self):
        """Test energy estimation through search space."""
        search_space = CoNASSearchSpace()
        arch_config = search_space.sample_architecture()
        
        energy = search_space.estimate_energy(arch_config)
        
        assert energy > 0
        assert isinstance(energy, float)
    
    def test_get_constraints(self):
        """Test constraint retrieval."""
        search_space = CoNASSearchSpace(
            num_layers=8,
            base_channels=128,
            max_skip_connections=10,
        )
        
        constraints = search_space.get_constraints()
        
        assert constraints["max_layers"] == 8
        assert constraints["base_channels"] == 128
        assert constraints["max_skip_connections"] == 10
        assert constraints["num_precision_levels"] == 4
        assert constraints["available_precisions"] == [32, 16, 8, 4]
    
    def test_device_compatibility(self):
        """Test that search space works on different devices."""
        # CPU test - use nn.Module's parameter iteration
        search_space_cpu = CoNASSearchSpace(num_layers=3)
        # Check that parameters are on CPU by accessing them directly
        all_cpu = all(
            param.device.type == "cpu" 
            for param in list(search_space_cpu.arch_params.values()) + 
                       list(search_space_cpu.precision_params.values())
        )
        assert all_cpu
        
        # CUDA test if available
        if torch.cuda.is_available():
            search_space_cuda = CoNASSearchSpace(num_layers=3).to("cuda")
            all_cuda = all(
                param.device.type == "cuda"
                for param in list(search_space_cuda.arch_params.values()) +
                           list(search_space_cuda.precision_params.values())
            )
            assert all_cuda
    
    def test_gradient_flow(self):
        """Test that gradients flow through architecture parameters."""
        search_space = CoNASSearchSpace(num_layers=3)
        
        # Get weights
        weights = search_space.get_architecture_weights()
        
        # Compute a simple loss
        loss = sum(w.sum() for w in weights.values())
        
        # Backpropagate
        loss.backward()
        
        # Check that gradients exist
        for param in search_space.arch_params.values():
            assert param.grad is not None
            assert param.grad.shape == param.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
