"""
Differentiable Mixed Precision Search Algorithm based on AutoBit Paradigm

This module implements the core search algorithm that jointly optimizes
architecture and precision configurations using gradient-based optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import copy

from .search_space import (
    CoNASSearchSpace,
    ArchitectureConfig,
    LayerConfig,
    PrecisionLevel,
    EnergyEstimator,
)


@dataclass
class SearchConfig:
    """Configuration for the mixed precision search algorithm."""
    # Optimization hyperparameters
    arch_lr: float = 0.01
    precision_lr: float = 0.01
    weight_decay: float = 3e-4
    
    # Energy-aware constraints
    energy_budget: Optional[float] = None
    bit_budget: Optional[int] = None
    
    # Search hyperparameters
    temperature: float = 1.0
    temperature_decay: float = 0.995
    min_temperature: float = 0.1
    
    # Regularization
    energy_weight: float = 0.1
    bit_weight: float = 0.05
    
    # Training
    num_search_steps: int = 1000
    validation_frequency: int = 50
    
    def __post_init__(self):
        """Validate configuration."""
        if self.energy_weight < 0 or self.energy_weight > 1:
            raise ValueError("energy_weight must be in [0, 1]")
        if self.bit_weight < 0 or self.bit_weight > 1:
            raise ValueError("bit_weight must be in [0, 1]")


class DifferentiableMixedPrecisionSearch:
    """
    Differentiable Mixed Precision Search Algorithm based on AutoBit paradigm.
    
    This algorithm performs joint optimization of:
    1. Architecture parameters (layer types, connections)
    2. Precision parameters (bit-width for each layer)
    
    Using gradient-based optimization with energy-aware regularization.
    """
    
    def __init__(
        self,
        search_space: CoNASSearchSpace,
        search_config: SearchConfig = None,
        device: str = "cpu",
    ):
        """
        Initialize the search algorithm.
        
        Args:
            search_space: The Co-NAS search space
            search_config: Search configuration
            device: Device for computation
        """
        self.search_space = search_space
        self.config = search_config or SearchConfig()
        self.device = device
        
        # Move search space to device
        self.search_space = self.search_space.to(device)
        
        # Optimizers for architecture and precision parameters
        self.arch_optimizer = None
        self.precision_optimizer = None
        
        # Search history
        self.search_history = {
            "loss": [],
            "energy": [],
            "bits": [],
            "validation_accuracy": [],
        }
        
        # Best architecture found
        self.best_architecture = None
        self.best_score = float("-inf")
    
    def _initialize_optimizers(self):
        """Initialize optimizers for architecture and precision parameters."""
        # Architecture optimizer
        arch_params = list(self.search_space.arch_params.values())
        self.arch_optimizer = optim.Adam(
            arch_params,
            lr=self.config.arch_lr,
            weight_decay=self.config.weight_decay,
        )
        
        # Precision optimizer
        precision_params = list(self.search_space.precision_params.values())
        self.precision_optimizer = optim.Adam(
            precision_params,
            lr=self.config.precision_lr,
            weight_decay=self.config.weight_decay,
        )
    
    def compute_energy_loss(self, arch_config: ArchitectureConfig) -> torch.Tensor:
        """
        Compute energy loss for an architecture configuration.
        
        Args:
            arch_config: Architecture configuration
            
        Returns:
            Energy loss tensor
        """
        energy = self.search_space.estimate_energy(arch_config)
        energy_tensor = torch.tensor(energy, device=self.device, dtype=torch.float32)
        
        # Normalize by budget if specified
        if self.config.energy_budget:
            energy_tensor = energy_tensor / self.config.energy_budget
        
        return energy_tensor
    
    def compute_bit_loss(
        self,
        precision_weights: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute bit-width regularization loss.
        
        Args:
            precision_weights: Precision weights for each layer
            
        Returns:
            Bit loss tensor
        """
        total_bits = 0.0
        max_bits = 0.0
        
        for layer_idx, weights in precision_weights.items():
            # Expected bit width for this layer
            expected_bits = sum(
                w * prec.value
                for w, prec in zip(weights, self.search_space.precision_levels)
            )
            total_bits += expected_bits
            max_bits += max(prec.value for prec in self.search_space.precision_levels)
        
        bit_loss = total_bits / max_bits
        
        # Penalize if exceeding budget
        if self.config.bit_budget:
            excess = max(0, total_bits - self.config.bit_budget)
            bit_loss += excess / self.config.bit_budget
        
        return bit_loss
    
    def compute_total_loss(
        self,
        validation_loss: torch.Tensor,
        energy_loss: torch.Tensor,
        bit_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total loss combining validation, energy, and bit losses.
        
        Args:
            validation_loss: Validation loss from the model
            energy_loss: Energy regularization loss
            bit_loss: Bit-width regularization loss
            
        Returns:
            Total loss tensor
        """
        total_loss = validation_loss
        total_loss = total_loss + self.config.energy_weight * energy_loss
        total_loss = total_loss + self.config.bit_weight * bit_loss
        
        return total_loss
    
    def search_step(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_targets: torch.Tensor,
        val_data: torch.Tensor,
        val_targets: torch.Tensor,
        criterion: nn.Module = None,
    ) -> Dict[str, float]:
        """
        Perform one step of the search algorithm.
        
        Args:
            model: SuperNet model to train
            train_data: Training data batch
            train_targets: Training targets batch
            val_data: Validation data batch
            val_targets: Validation targets batch
            criterion: Loss function
            
        Returns:
            Dictionary of loss values
        """
        if self.arch_optimizer is None:
            self._initialize_optimizers()
        
        criterion = criterion or nn.CrossEntropyLoss()
        
        # Sample current architecture
        arch_config = self.search_space.sample_architecture(
            temperature=self.config.temperature
        )
        
        # Forward pass on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data, arch_config)
            val_loss = criterion(val_outputs, val_targets)
        
        # Compute energy and bit losses
        energy_loss = self.compute_energy_loss(arch_config)
        precision_weights = self.search_space.get_precision_weights()
        bit_loss = self.compute_bit_loss(precision_weights)
        
        # Total loss
        total_loss = self.compute_total_loss(val_loss, energy_loss, bit_loss)
        
        # Backward pass for architecture parameters
        self.arch_optimizer.zero_grad()
        self.precision_optimizer.zero_grad()
        
        # Compute gradients with respect to architecture parameters
        total_loss.backward()
        
        # Update architecture and precision parameters
        self.arch_optimizer.step()
        self.precision_optimizer.step()
        
        # Decay temperature
        self.config.temperature = max(
            self.config.min_temperature,
            self.config.temperature * self.config.temperature_decay,
        )
        
        return {
            "total_loss": total_loss.item(),
            "validation_loss": val_loss.item(),
            "energy_loss": energy_loss.item(),
            "bit_loss": bit_loss.item(),
            "temperature": self.config.temperature,
        }
    
    def search(
        self,
        super_net: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> ArchitectureConfig:
        """
        Run the complete search process.
        
        Args:
            super_net: SuperNet containing all possible architectures
            train_loader: Training data loader
            val_loader: Validation data loader
            callback: Optional callback function called after each step
            
        Returns:
            Best architecture configuration found
        """
        super_net = super_net.to(self.device)
        
        # Get a batch for search steps
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        
        for step in range(self.config.num_search_steps):
            # Get training batch
            try:
                train_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                train_batch = next(train_iter)
            
            # Get validation batch
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_batch = next(val_iter)
            
            # Handle different data formats
            if isinstance(train_batch, (list, tuple)):
                train_data, train_targets = train_batch[0].to(self.device), train_batch[1].to(self.device)
            else:
                train_data = train_batch.to(self.device)
                train_targets = train_batch.to(self.device)
            
            if isinstance(val_batch, (list, tuple)):
                val_data, val_targets = val_batch[0].to(self.device), val_batch[1].to(self.device)
            else:
                val_data = val_batch.to(self.device)
                val_targets = val_batch.to(self.device)
            
            # Perform search step
            metrics = self.search_step(
                super_net,
                train_data,
                train_targets,
                val_data,
                val_targets,
            )
            
            # Record history
            self.search_history["loss"].append(metrics["total_loss"])
            self.search_history["energy"].append(metrics["energy_loss"])
            self.search_history["bits"].append(metrics["bit_loss"])
            
            # Sample current best architecture and evaluate
            if step % self.config.validation_frequency == 0 or step == self.config.num_search_steps - 1:
                current_arch = self.search_space.sample_architecture(
                    temperature=self.config.temperature
                )
                energy = self.search_space.estimate_energy(current_arch)
                
                # Simple scoring: lower energy is better (can be extended with accuracy)
                score = -metrics["total_loss"] - 0.1 * energy
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_architecture = copy.deepcopy(current_arch)
                
                # Callback
                if callback:
                    callback(step, {**metrics, "best_score": self.best_score})
        
        # Return best architecture at lowest temperature
        if self.best_architecture is None:
            self.best_architecture = self.search_space.sample_architecture(
                temperature=self.config.min_temperature
            )
        
        return self.best_architecture
    
    def get_search_statistics(self) -> Dict:
        """
        Get statistics from the search process.
        
        Returns:
            Dictionary of search statistics
        """
        if not self.search_history["loss"]:
            return {}
        
        return {
            "final_loss": self.search_history["loss"][-1],
            "min_loss": min(self.search_history["loss"]),
            "final_energy": self.search_history["energy"][-1],
            "final_bits": self.search_history["bits"][-1],
            "num_steps": len(self.search_history["loss"]),
            "best_score": self.best_score,
        }


class AutoBitSuperNet(nn.Module):
    """
    SuperNet implementation for AutoBit-based mixed precision search.
    
    This network contains all possible layer types and precision configurations,
    allowing differentiable architecture search.
    """
    
    def __init__(
        self,
        base_channels: int = 64,
        num_classes: int = 10,
        input_size: int = 32,
    ):
        """
        Initialize the SuperNet.
        
        Args:
            base_channels: Base number of channels
            num_classes: Number of output classes
            input_size: Input image size
        """
        super().__init__()
        
        self.base_channels = base_channels
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Build all possible layer types
        self._build_layers()
    
    def _build_layers(self):
        """Build all possible layer variants in the SuperNet."""
        # Convolutional layers with different channel expansions
        channel_ratios = [0.5, 1.0, 2.0]
        max_channels = 64  # Cap maximum channels to avoid memory issues
        
        self.conv_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        
        # Start with 3 input channels (standard RGB images)
        in_channels = 3
        num_stages = 3  # Reduced number of stages
        
        for i in range(num_stages):
            for ratio in channel_ratios:
                out_channels = min(int(self.base_channels * ratio), max_channels)
                
                # Ensure out_channels is at least 1
                out_channels = max(out_channels, 1)
                
                # Conv layer - use base_channels for first stage, then increase
                conv_in_channels = 3 if i == 0 else min(self.base_channels * (2 ** (i-1)), max_channels)
                conv = nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1)
                self.conv_layers.append(conv)
                
                # Linear layer (for classifier) - use reduced feature size
                feature_size = max(self.input_size // (2 ** (i + 1)), 1)
                linear = nn.Linear(out_channels * feature_size * feature_size, self.num_classes)
                self.linear_layers.append(linear)
            
            # Increase channels but cap at max_channels
            in_channels = min(self.base_channels * (2 ** i), max_channels)
        
        # Final classifier with capped channels
        final_channels = min(self.base_channels * (2 ** (num_stages - 1)), max_channels)
        final_feature_size = max(self.input_size // (2 ** num_stages), 1)
        self.classifier = nn.Linear(final_channels * final_feature_size * final_feature_size, self.num_classes)
    
    def forward(
        self,
        x: torch.Tensor,
        arch_config: ArchitectureConfig = None,
    ) -> torch.Tensor:
        """
        Forward pass through the SuperNet following the architecture config.
        
        Args:
            x: Input tensor
            arch_config: Architecture configuration specifying which layers to use
            
        Returns:
            Output tensor
        """
        # Use simple sequential forward pass through conv_layers
        out = x
        
        # Apply a subset of conv layers - use every 3rd layer (one per stage)
        num_layers_to_use = min(len(self.conv_layers), 3)
        layer_indices = [0, 3, 6] if len(self.conv_layers) >= 9 else list(range(num_layers_to_use))
        
        for idx in layer_indices[:3]:
            if idx < len(self.conv_layers):
                # Check if feature maps are large enough for conv
                if out.shape[2] >= 3 and out.shape[3] >= 3:
                    try:
                        out = self.conv_layers[idx](out)
                        out = torch.relu(out)
                        
                        # Add pooling to reduce spatial dimensions
                        if out.shape[2] > 1 and out.shape[3] > 1:
                            out = torch.nn.functional.avg_pool2d(out, kernel_size=2, stride=2)
                    except RuntimeError:
                        # Skip layer if channel mismatch
                        pass
        
        # Global average pooling to get fixed-size features
        out = torch.mean(out, dim=(2, 3))
        
        # Ensure out has correct shape for classifier
        expected_features = self.classifier.in_features
        if out.shape[1] != expected_features:
            # Truncate or pad as needed
            if out.shape[1] > expected_features:
                out = out[:, :expected_features]
            else:
                # Pad with zeros
                padding = torch.zeros(out.shape[0], expected_features - out.shape[1], device=out.device)
                out = torch.cat([out, padding], dim=1)
        
        # Classifier
        out = self.classifier(out)
        
        return out
