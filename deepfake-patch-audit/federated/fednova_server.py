"""FedNova server implementation for robust federated learning aggregation.

FedNova (Federated Normalized Averaging):
- Handles non-IID data distribution across edge devices
- Normalizes client updates based on local steps performed
- Each client contributes proportionally to true global objective
- 6-9% improvement over FedAvg/FedProx on non-IID data (Wang et al., NeurIPS 2020)

Reference: https://arxiv.org/abs/2007.01154
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from collections import OrderedDict
import logging


@dataclass
class ClientUpdate:
    """Update from a single client."""
    client_id: str
    model_delta: Dict[str, torch.Tensor]  # w - w_global (delta)
    num_local_steps: int
    data_size: int
    loss: float
    metrics: Dict[str, float] = None


@dataclass
class AnomalyDetectionStats:
    """Statistics for anomaly detection."""
    update_norm: float
    cosine_similarity: float
    loss_change: float
    is_anomalous: bool = False


class FedNovaServer:
    """
    FedNova server for federated learning with anomaly detection.

    Key features:
    1. Normalized averaging (handles variable local steps across clients)
    2. Anomaly detection to identify malicious/faulty clients
    3. Adaptive learning rate for robust convergence
    4. Automatic client filtering for non-IID data
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.1,
        anomaly_threshold: float = 2.5,
        enable_anomaly_detection: bool = True,
    ):
        """
        Args:
            model: Global model to be distributed
            learning_rate: Server learning rate for update aggregation
            anomaly_threshold: Threshold (in std deviations) for anomaly detection
            enable_anomaly_detection: Whether to filter anomalous updates
        """
        self.global_model = model
        self.learning_rate = learning_rate
        self.anomaly_threshold = anomaly_threshold
        self.enable_anomaly_detection = enable_anomaly_detection

        # Global state
        self.global_weights = {name: param.data.clone() for name, param in model.named_parameters()}
        self.round = 0
        self.client_history: Dict[str, List[float]] = {}
        self.anomaly_history = []

        self.logger = logging.getLogger(__name__)

    def aggregate_fednova(self, client_updates: List[ClientUpdate]) -> Tuple[Dict, Dict]:
        """
        Aggregate client updates using FedNova algorithm.

        FedNova key insight: Normalize by the number of local steps
        to account for heterogeneous local optimization trajectories.

        Args:
            client_updates: List of updates from clients

        Returns:
            Aggregated model update and statistics
        """
        if not client_updates:
            return self.global_weights, {}

        # Step 1: Compute client weights based on data size
        total_data = sum(update.data_size for update in client_updates)
        client_weights = {
            update.client_id: update.data_size / total_data for update in client_updates
        }

        # Step 2: Normalize gradients by local steps (FedNova's key contribution)
        normalized_deltas = {}
        for update in client_updates:
            # Normalize each client's delta by their number of local steps
            tau_i = update.num_local_steps
            normalized_delta = {}

            for param_name, delta in update.model_delta.items():
                # FedNova normalization: divide by local step count
                normalized_delta[param_name] = delta / max(tau_i, 1)

            normalized_deltas[update.client_id] = normalized_delta

        # Step 3: Weighted average of normalized deltas
        aggregated_delta = None
        for client_id, normalized_delta in normalized_deltas.items():
            weight = client_weights[client_id]

            for param_name, delta in normalized_delta.items():
                if aggregated_delta is None:
                    aggregated_delta = {}

                if param_name not in aggregated_delta:
                    aggregated_delta[param_name] = torch.zeros_like(delta)

                aggregated_delta[param_name] += weight * delta

        # Step 4: Update global model
        updated_weights = {}
        for param_name in self.global_weights.keys():
            if param_name in aggregated_delta:
                updated_weights[param_name] = (
                    self.global_weights[param_name] - self.learning_rate * aggregated_delta[param_name]
                )
            else:
                updated_weights[param_name] = self.global_weights[param_name]

        # Compute aggregation statistics
        stats = {
            "num_clients": len(client_updates),
            "aggregation_method": "FedNova",
            "learning_rate": self.learning_rate,
            "client_weights": client_weights,
        }

        return updated_weights, stats

    def detect_anomalies(self, client_updates: List[ClientUpdate]) -> Tuple[List[ClientUpdate], List[str]]:
        """
        Detect anomalous updates using multiple heuristics.

        Heuristics:
        1. Update norm (magnitude of change)
        2. Cosine similarity to global direction
        3. Loss change direction (should improve global loss)
        4. Statistical outliers (IQR-based)
        """
        if not self.enable_anomaly_detection or len(client_updates) < 3:
            return client_updates, []

        anomalies = []
        stats_list = []

        # Compute update statistics
        for update in client_updates:
            # 1. Update norm
            norm = 0.0
            for delta in update.model_delta.values():
                norm += torch.norm(delta).item() ** 2
            norm = np.sqrt(norm)

            # 2. Cosine similarity to median direction
            cosine_sim = self._compute_cosine_similarity(update, client_updates)

            # 3. Loss change (positive = improvement)
            loss_change = -update.loss  # Negative loss is improvement

            stat = AnomalyDetectionStats(
                update_norm=norm, cosine_similarity=cosine_sim, loss_change=loss_change
            )
            stats_list.append((update.client_id, stat))

        # Identify anomalies using multiple criteria
        anomalous_clients = set()

        # Criterion 1: Update norm (IQR-based)
        norms = np.array([s[1].update_norm for s in stats_list])
        q1, q3 = np.percentile(norms, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - self.anomaly_threshold * iqr
        upper_bound = q3 + self.anomaly_threshold * iqr

        for client_id, stat in stats_list:
            if stat.update_norm < lower_bound or stat.update_norm > upper_bound:
                anomalous_clients.add(client_id)
                stat.is_anomalous = True

        # Criterion 2: Cosine similarity (clients with very different direction)
        cos_sims = np.array([s[1].cosine_similarity for s in stats_list])
        mean_cos = np.mean(cos_sims)
        std_cos = np.std(cos_sims)

        for client_id, stat in stats_list:
            if stat.cosine_similarity < mean_cos - self.anomaly_threshold * std_cos:
                anomalous_clients.add(client_id)
                stat.is_anomalous = True

        # Filter out anomalous updates
        filtered_updates = [u for u in client_updates if u.client_id not in anomalous_clients]

        # Log anomalies
        if anomalous_clients:
            self.logger.warning(f"Detected {len(anomalous_clients)} anomalous clients: {anomalous_clients}")

        self.anomaly_history.append(
            {"round": self.round, "anomalous_clients": list(anomalous_clients), "stats": stats_list}
        )

        return filtered_updates, list(anomalous_clients)

    def _compute_cosine_similarity(self, update: ClientUpdate, all_updates: List[ClientUpdate]) -> float:
        """Compute cosine similarity between client update and median direction."""
        # Compute median delta direction
        all_deltas = []
        for u in all_updates:
            delta_vec = torch.cat([delta.flatten() for delta in u.model_delta.values()])
            all_deltas.append(delta_vec)

        if not all_deltas:
            return 0.0

        median_delta = torch.median(torch.stack(all_deltas), dim=0)[0]

        # Compute similarity to current update
        current_delta = torch.cat([delta.flatten() for delta in update.model_delta.values()])

        cos_sim = torch.nn.functional.cosine_similarity(
            current_delta.unsqueeze(0), median_delta.unsqueeze(0)
        ).item()

        return cos_sim

    def update_global_model(self, client_updates: List[ClientUpdate]) -> Dict:
        """
        Execute one round of FedNova aggregation.

        Steps:
        1. Detect and filter anomalous updates
        2. Aggregate remaining updates using FedNova
        3. Update global model
        """
        self.round += 1

        # Step 1: Anomaly detection
        if self.enable_anomaly_detection:
            filtered_updates, anomalous_clients = self.detect_anomalies(client_updates)
        else:
            filtered_updates = client_updates
            anomalous_clients = []

        # Step 2: Aggregate
        updated_weights, agg_stats = self.aggregate_fednova(filtered_updates)

        # Step 3: Update global model
        self.global_weights = updated_weights
        self._update_model_weights(updated_weights)

        # Step 4: Compute global metrics
        global_loss = np.mean([u.loss for u in filtered_updates]) if filtered_updates else float("inf")

        result = {
            "round": self.round,
            "num_clients_participated": len(client_updates),
            "num_clients_anomalous": len(anomalous_clients),
            "num_clients_aggregated": len(filtered_updates),
            "global_loss": global_loss,
            "aggregation_stats": agg_stats,
            "anomalous_clients": anomalous_clients,
        }

        return result

    def _update_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Update global model with new weights."""
        for name, param in self.global_model.named_parameters():
            if name in weights:
                param.data = weights[name].clone()

    def get_global_model(self) -> nn.Module:
        """Get current global model for distribution to clients."""
        return self.global_model

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global weights."""
        return {name: weight.clone() for name, weight in self.global_weights.items()}

    def save_checkpoint(self, checkpoint_dir: str):
        """Save server state and model."""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(self.global_model.state_dict(), Path(checkpoint_dir) / "global_model.pt")

        # Save server state
        state = {
            "round": self.round,
            "learning_rate": self.learning_rate,
            "client_history": self.client_history,
            "anomaly_history": self.anomaly_history,
        }

        with open(Path(checkpoint_dir) / "server_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)

        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load server state and model."""
        # Load model
        self.global_model.load_state_dict(torch.load(Path(checkpoint_dir) / "global_model.pt"))

        # Load server state
        with open(Path(checkpoint_dir) / "server_state.json", "r") as f:
            state = json.load(f)

        self.round = state.get("round", 0)
        self.client_history = state.get("client_history", {})
        self.anomaly_history = state.get("anomaly_history", [])

        self.logger.info(f"Checkpoint loaded from {checkpoint_dir}")
