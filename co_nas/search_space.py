"""
Novel Energy-Aware Co-NAS Joint Search Space

This module defines the joint search space for architecture and mixed-precision
configurations based on the AutoBit paradigm with energy-aware considerations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PrecisionLevel(Enum):
    """Supported precision levels for mixed-precision quantization."""
    FP32 = 32
    FP16 = 16
    INT8 = 8
    INT4 = 4
    INT2 = 2


@dataclass
class LayerConfig:
    """Configuration for a single layer in the search space."""
    layer_type: str
    in_features: int
    out_features: int
    precision: PrecisionLevel
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    activation: str = "relu"
    
    def get_bit_width(self) -> int:
        """Return the bit width for this layer's precision."""
        return self.precision.value


@dataclass
class ArchitectureConfig:
    """Complete architecture configuration within the joint search space."""
    layers: List[LayerConfig]
    skip_connections: List[Tuple[int, int]]  # (from_layer, to_layer)
    
    def total_bits(self) -> int:
        """Calculate total bits required for the architecture."""
        total = 0
        for layer in self.layers:
            if layer.layer_type == "conv":
                params = layer.kernel_size * layer.kernel_size * layer.in_features * layer.out_features
            elif layer.layer_type == "linear":
                params = layer.in_features * layer.out_features
            else:
                params = layer.in_features * layer.out_features
            
            total += params * layer.get_bit_width()
        return total
    
    def num_layers(self) -> int:
        """Return number of layers in the architecture."""
        return len(self.layers)


class EnergyEstimator:
    """
    Energy estimation model for different operations and precisions.
    Based on hardware-aware energy costs.
    """
    
    # Energy costs in pJ per operation (approximate values)
    ENERGY_COSTS = {
        PrecisionLevel.FP32: {"mac": 3.7, "memory": 0.5},
        PrecisionLevel.FP16: {"mac": 1.1, "memory": 0.2},
        PrecisionLevel.INT8: {"mac": 0.4, "memory": 0.1},
        PrecisionLevel.INT4: {"mac": 0.2, "memory": 0.05},
        PrecisionLevel.INT2: {"mac": 0.1, "memory": 0.03},
    }
    
    @classmethod
    def estimate_layer_energy(cls, config: LayerConfig, num_operations: int) -> float:
        """
        Estimate energy consumption for a layer.
        
        Args:
            config: Layer configuration
            num_operations: Number of MAC operations
            
        Returns:
            Estimated energy in pJ
        """
        costs = cls.ENERGY_COSTS.get(config.precision, cls.ENERGY_COSTS[PrecisionLevel.FP32])
        mac_energy = costs["mac"] * num_operations
        memory_energy = costs["memory"] * num_operations
        
        return mac_energy + memory_energy
    
    @classmethod
    def estimate_architecture_energy(cls, arch_config: ArchitectureConfig) -> float:
        """
        Estimate total energy consumption for an architecture.
        
        Args:
            arch_config: Architecture configuration
            
        Returns:
            Total estimated energy in pJ
        """
        total_energy = 0.0
        
        for layer in arch_config.layers:
            if layer.layer_type == "conv":
                num_ops = (layer.kernel_size ** 2) * layer.in_features * layer.out_features
            elif layer.layer_type == "linear":
                num_ops = layer.in_features * layer.out_features
            else:
                num_ops = layer.in_features * layer.out_features
            
            total_energy += cls.estimate_layer_energy(layer, num_ops)
        
        return total_energy


class CoNASSearchSpace(nn.Module):
    """
    Novel Energy-Aware Co-NAS Joint Search Space.
    
    This class implements a differentiable search space that jointly optimizes
    architecture parameters and mixed-precision configurations based on the
    AutoBit paradigm with energy-aware constraints.
    """
    
    def __init__(
        self,
        num_layers: int = 10,
        base_channels: int = 64,
        precision_levels: List[PrecisionLevel] = None,
        max_skip_connections: int = 5,
    ):
        """
        Initialize the Co-NAS search space.
        
        Args:
            num_layers: Maximum number of layers to consider
            base_channels: Base number of channels
            precision_levels: Available precision levels for search
            max_skip_connections: Maximum number of skip connections
        """
        super().__init__()  # Call parent Module init
        
        self.num_layers = num_layers
        self.base_channels = base_channels
        self.precision_levels = precision_levels or [
            PrecisionLevel.FP32,
            PrecisionLevel.FP16,
            PrecisionLevel.INT8,
            PrecisionLevel.INT4,
        ]
        self.max_skip_connections = max_skip_connections
        
        # Architecture parameters (differentiable)
        self.arch_params = nn.ParameterDict()
        self._initialize_arch_params()
        
        # Precision parameters (differentiable relaxation)
        self.precision_params = nn.ParameterDict()
        self._initialize_precision_params()
        
        # Energy estimator
        self.energy_estimator = EnergyEstimator()
    
    def _initialize_arch_params(self):
        """Initialize differentiable architecture parameters."""
        # Layer type probabilities
        layer_types = ["conv", "linear", "pool", "identity"]
        for i in range(self.num_layers):
            self.arch_params[f"layer_type_{i}"] = nn.Parameter(
                torch.randn(len(layer_types)) / 10.0
            )
        
        # Channel expansion ratios
        for i in range(self.num_layers):
            self.arch_params[f"channels_{i}"] = nn.Parameter(
                torch.randn(4) / 10.0  # [0.5x, 1x, 2x, 4x]
            )
        
        # Skip connection probabilities
        self.arch_params["skip_connections"] = nn.Parameter(
            torch.randn(self.num_layers, self.num_layers) / 10.0
        )
    
    def _initialize_precision_params(self):
        """Initialize differentiable precision parameters using softmax relaxation."""
        num_precisions = len(self.precision_levels)
        
        for i in range(self.num_layers):
            # Precision weights for each layer (will be softmaxed)
            self.precision_params[f"layer_{i}"] = nn.Parameter(
                torch.randn(num_precisions) / 10.0
            )
    
    def get_architecture_weights(self) -> Dict[str, torch.Tensor]:
        """Get softmax-normalized architecture weights."""
        weights = {}
        
        layer_types = ["conv", "linear", "pool", "identity"]
        channel_ratios = [0.5, 1.0, 2.0, 4.0]
        
        for i in range(self.num_layers):
            weights[f"layer_type_{i}"] = torch.softmax(
                self.arch_params[f"layer_type_{i}"], dim=-1
            )
            weights[f"channels_{i}"] = torch.softmax(
                self.arch_params[f"channels_{i}"], dim=-1
            )
        
        weights["skip_connections"] = torch.sigmoid(
            self.arch_params["skip_connections"]
        )
        
        return weights
    
    def get_precision_weights(self) -> Dict[int, torch.Tensor]:
        """Get softmax-normalized precision weights for each layer."""
        weights = {}
        
        for i in range(self.num_layers):
            weights[i] = torch.softmax(
                self.precision_params[f"layer_{i}"], dim=-1
            )
        
        return weights
    
    def sample_architecture(self, temperature: float = 1.0) -> ArchitectureConfig:
        """
        Sample a discrete architecture from the differentiable search space.
        
        Args:
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Sampled ArchitectureConfig
        """
        layer_types = ["conv", "linear", "pool", "identity"]
        channel_ratios = [0.5, 1.0, 2.0, 4.0]
        
        layers = []
        arch_weights = self.get_architecture_weights()
        precision_weights = self.get_precision_weights()
        
        in_channels = self.base_channels
        
        for i in range(self.num_layers):
            # Sample layer type
            type_probs = arch_weights[f"layer_type_{i}"] / temperature
            layer_type_idx = torch.argmax(type_probs).item()
            layer_type = layer_types[layer_type_idx]
            
            # Sample channel ratio
            channel_probs = arch_weights[f"channels_{i}"] / temperature
            channel_idx = torch.argmax(channel_probs).item()
            out_channels = int(in_channels * channel_ratios[channel_idx])
            
            # Sample precision
            prec_probs = precision_weights[i] / temperature
            prec_idx = torch.argmax(prec_probs).item()
            precision = self.precision_levels[prec_idx]
            
            # Create layer config
            layer_config = LayerConfig(
                layer_type=layer_type,
                in_features=in_channels,
                out_features=out_channels,
                precision=precision,
                kernel_size=3 if layer_type == "conv" else None,
                stride=1 if layer_type == "conv" else None,
            )
            
            layers.append(layer_config)
            in_channels = out_channels
        
        # Sample skip connections
        skip_mask = arch_weights["skip_connections"] > 0.5
        skip_connections = []
        for i in range(self.num_layers):
            for j in range(i + 2, min(i + 4, self.num_layers)):
                if skip_mask[i, j]:
                    skip_connections.append((i, j))
                    if len(skip_connections) >= self.max_skip_connections:
                        break
            if len(skip_connections) >= self.max_skip_connections:
                break
        
        return ArchitectureConfig(layers=layers, skip_connections=skip_connections)
    
    def estimate_energy(self, arch_config: ArchitectureConfig) -> float:
        """
        Estimate energy consumption for a given architecture.
        
        Args:
            arch_config: Architecture configuration
            
        Returns:
            Estimated energy in pJ
        """
        return self.energy_estimator.estimate_architecture_energy(arch_config)
    
    def get_constraints(self) -> Dict[str, float]:
        """
        Get current search space constraints.
        
        Returns:
            Dictionary of constraint values
        """
        return {
            "max_layers": self.num_layers,
            "base_channels": self.base_channels,
            "num_precision_levels": len(self.precision_levels),
            "max_skip_connections": self.max_skip_connections,
            "available_precisions": [p.value for p in self.precision_levels],
        }
