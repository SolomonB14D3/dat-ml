"""
Topological Stability Loss Functions.

Implements loss functions derived from DAT and H3-Hybrid principles:
- Phason localization loss (encourages 4.2x improvement)
- Energy depth loss (biases toward metastable minimum)
- Coordination loss (enforces H3 Hybrid coordination patterns)
- Manifold consistency loss (preserves H3 structure)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

from ..transforms.e6_projection import TAU


class PhasonLocalizationLoss(nn.Module):
    """
    Penalizes deviations from target phason localization.

    From DAT: The phason-transistor effect achieves ~4.2x improved
    topological localization compared to cubic lattices.
    """

    def __init__(
        self,
        target_improvement: float = 4.2,
        weight: float = 0.1
    ):
        super().__init__()
        self.target = target_improvement
        self.weight = weight

    def forward(
        self,
        parallel: torch.Tensor,
        perp: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute phason localization loss.

        Args:
            parallel: Parallel (physical) space features
            perp: Perpendicular (phason) space features

        Returns:
            Loss encouraging target localization ratio
        """
        # Variance in each space
        var_parallel = torch.var(parallel, dim=-1, keepdim=True)
        var_perp = torch.var(perp, dim=-1, keepdim=True) + 1e-8

        # Localization ratio
        localization = var_parallel / var_perp

        # Loss: penalize deviation from target
        loss = F.mse_loss(localization, torch.full_like(localization, self.target))

        return self.weight * loss


class EnergyDepthLoss(nn.Module):
    """
    Biases model toward deep metastable energy minima.

    From H3-Hybrid: The hybrid phase achieves -7.68ε/atom vs
    -6.48ε/atom for equilibrium phases.
    """

    def __init__(
        self,
        target_energy: float = -7.68,
        baseline_energy: float = -6.48,
        weight: float = 0.1
    ):
        super().__init__()
        self.target = target_energy
        self.baseline = baseline_energy
        self.weight = weight
        self.improvement_ratio = target_energy / baseline_energy

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        """
        Compute energy depth loss.

        Args:
            energy: Predicted energy values

        Returns:
            Loss encouraging deep energy states
        """
        # Normalize energy to [0, 1] range where 0 is target, 1 is baseline
        normalized = (energy - self.target) / (self.baseline - self.target)

        # Penalize being above target (positive normalized values)
        loss = F.relu(normalized).mean()

        return self.weight * loss


class CoordinationLoss(nn.Module):
    """
    Enforces H3 Hybrid coordination patterns.

    From H3-Hybrid: RDF peak at 1.081σ indicates highly compressed
    coordination shells.
    """

    def __init__(
        self,
        target_coordination: float = 1.081,
        weight: float = 0.1
    ):
        super().__init__()
        self.target = target_coordination
        self.weight = weight

    def forward(self, shell_occupancy: torch.Tensor) -> torch.Tensor:
        """
        Compute coordination loss.

        Args:
            shell_occupancy: Normalized shell occupancy (batch, shells)

        Returns:
            Loss encouraging target coordination pattern
        """
        # First shell should have highest occupancy (compressed coordination)
        first_shell = shell_occupancy[:, 0]

        # Target: normalized occupancy near 1.0 (full coordination)
        loss = F.mse_loss(first_shell, torch.ones_like(first_shell))

        # Penalize occupancy in outer shells (should be lower)
        if shell_occupancy.shape[1] > 1:
            outer_shells = shell_occupancy[:, 1:]
            outer_loss = (outer_shells ** 2).mean()
            loss = loss + 0.5 * outer_loss

        return self.weight * loss


class ManifoldConsistencyLoss(nn.Module):
    """
    Preserves H3 manifold structure in learned representations.

    Enforces:
    - Icosahedral symmetry relationships
    - Golden ratio scaling invariance
    - Quasicrystal aperiodicity patterns
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        self.tau = TAU

    def forward(
        self,
        features: torch.Tensor,
        intermediates: Optional[list] = None
    ) -> torch.Tensor:
        """
        Compute manifold consistency loss.

        Args:
            features: Current feature representations
            intermediates: Optional list of intermediate representations

        Returns:
            Loss preserving manifold structure
        """
        loss = torch.tensor(0.0, device=features.device)

        # Golden ratio scaling consistency
        # Features should be approximately self-similar under τ scaling
        scaled = features * self.tau
        scale_diff = torch.abs(features.std() * self.tau - scaled.std())
        loss = loss + scale_diff

        # If we have intermediates, check progression consistency
        if intermediates and len(intermediates) > 1:
            # Compression should follow golden ratio
            for i in range(len(intermediates) - 1):
                curr_norm = intermediates[i].norm(dim=-1).mean()
                next_norm = intermediates[i + 1].norm(dim=-1).mean()

                # Ratio should be close to τ
                ratio = curr_norm / (next_norm + 1e-8)
                ratio_loss = (ratio - self.tau) ** 2
                loss = loss + 0.1 * ratio_loss

        return self.weight * loss


class IcosahedralSymmetryLoss(nn.Module):
    """
    Enforces 5-fold and 3-fold rotational symmetry patterns.
    """

    def __init__(self, weight: float = 0.05):
        super().__init__()
        self.weight = weight
        # 5-fold rotation angle
        self.angle_5 = 2 * math.pi / 5
        # 3-fold rotation angle
        self.angle_3 = 2 * math.pi / 3

    def _rotate_2d(self, x: torch.Tensor, angle: float) -> torch.Tensor:
        """Apply 2D rotation to last two dimensions."""
        c, s = math.cos(angle), math.sin(angle)
        x_rot = x.clone()
        x_rot[..., -2] = c * x[..., -2] - s * x[..., -1]
        x_rot[..., -1] = s * x[..., -2] + c * x[..., -1]
        return x_rot

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetry consistency loss.

        Features should have similar statistics under icosahedral rotations.
        """
        if features.shape[-1] < 2:
            return torch.tensor(0.0, device=features.device)

        loss = torch.tensor(0.0, device=features.device)

        # 5-fold symmetry check
        rotated_5 = self._rotate_2d(features, self.angle_5)
        loss = loss + F.mse_loss(features.var(dim=0), rotated_5.var(dim=0))

        # 3-fold symmetry check
        rotated_3 = self._rotate_2d(features, self.angle_3)
        loss = loss + F.mse_loss(features.var(dim=0), rotated_3.var(dim=0))

        return self.weight * loss


class StabilityAnnealingLoss(nn.Module):
    """
    Loss that anneals from exploration to stability.

    Inspired by H3-Hybrid thermal annealing from 0.2K to 0.0K
    that validates phase stability.
    """

    def __init__(
        self,
        start_temp: float = 0.2,
        end_temp: float = 0.01,
        total_steps: int = 10000,
        weight: float = 0.1
    ):
        super().__init__()
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_steps = total_steps
        self.weight = weight
        self.register_buffer('current_step', torch.tensor(0))

    def get_temperature(self) -> float:
        """Get current annealing temperature."""
        progress = min(self.current_step.item() / self.total_steps, 1.0)
        temp = self.start_temp * (1 - progress) + self.end_temp * progress
        return temp

    def step(self):
        """Increment annealing step."""
        self.current_step += 1

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        """
        Compute stability loss with temperature-dependent penalty.

        At high temperature: allow exploration (weak stability penalty)
        At low temperature: enforce strict stability (strong penalty)
        """
        temp = self.get_temperature()

        # Energy fluctuation penalty, scaled by inverse temperature
        fluctuation = energy.var()
        loss = fluctuation / (temp + 1e-8)

        return self.weight * loss


class CombinedTopologicalLoss(nn.Module):
    """
    Combined loss function incorporating all topological constraints.
    """

    def __init__(
        self,
        prediction_weight: float = 1.0,
        localization_weight: float = 0.1,
        energy_weight: float = 0.1,
        coordination_weight: float = 0.1,
        manifold_weight: float = 0.05,
        symmetry_weight: float = 0.05,
        use_annealing: bool = True
    ):
        super().__init__()
        self.prediction_weight = prediction_weight

        self.localization = PhasonLocalizationLoss(weight=localization_weight)
        self.energy_depth = EnergyDepthLoss(weight=energy_weight)
        self.coordination = CoordinationLoss(weight=coordination_weight)
        self.manifold = ManifoldConsistencyLoss(weight=manifold_weight)
        self.symmetry = IcosahedralSymmetryLoss(weight=symmetry_weight)

        self.use_annealing = use_annealing
        if use_annealing:
            self.annealing = StabilityAnnealingLoss(weight=0.05)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        model_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined topological loss.

        Args:
            prediction: Model predictions
            target: Ground truth targets
            model_output: Full model output dict with intermediate values

        Returns:
            Dict with total loss and component breakdowns
        """
        losses = {}

        # Primary prediction loss
        pred_loss = F.mse_loss(prediction.squeeze(), target.squeeze())
        losses['prediction'] = self.prediction_weight * pred_loss

        # Topological losses
        parallel = model_output.get('parallel')
        perp = model_output.get('perp')
        if parallel is not None and perp is not None:
            losses['localization'] = self.localization(parallel, perp)

        if 'energy' in model_output and model_output['energy'] is not None:
            losses['energy_depth'] = self.energy_depth(model_output['energy'])

            if self.use_annealing:
                losses['annealing'] = self.annealing(model_output['energy'])
                self.annealing.step()

        if 'shell_occupancy' in model_output and model_output['shell_occupancy'] is not None:
            losses['coordination'] = self.coordination(model_output['shell_occupancy'])

        # Manifold and symmetry losses on features
        features = model_output.get('features', prediction)
        losses['manifold'] = self.manifold(features)
        losses['symmetry'] = self.symmetry(features)

        # Total loss
        losses['total'] = sum(losses.values())

        return losses
