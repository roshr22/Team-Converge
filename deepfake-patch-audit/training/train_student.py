"""Student model distillation training."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json


class StudentTrainer:
    """
    Train student model with knowledge distillation from teacher.
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        train_loader,
        val_loader,
        criterion,
        device="cuda",
        lr=0.001,
        weight_decay=1e-4,
    ):
        """
        Args:
            student_model: Student model to train
            teacher_model: Pretrained teacher model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function (DistillationLoss)
            device: Device for training
            lr: Learning rate
            weight_decay: Weight decay
        """
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device

        self.teacher_model.eval()  # Freeze teacher

        self.optimizer = optim.Adam(
            self.student_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def train_epoch(self):
        """Train for one epoch."""
        self.student_model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Student forward pass
            student_logits = self.student_model(images)

            # Teacher forward pass (no grad)
            with torch.no_grad():
                teacher_logits = self.teacher_model(images)

            # Compute loss
            loss = self.criterion(student_logits, teacher_logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.history["train_loss"].append(avg_loss)
        return avg_loss

    def validate(self):
        """Validate on validation set."""
        self.student_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                student_logits = self.student_model(images)
                teacher_logits = self.teacher_model(images)

                loss = self.criterion(student_logits, teacher_logits, labels)
                total_loss += loss.item()

                # Accuracy
                _, predicted = torch.max(student_logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        self.history["val_loss"].append(avg_loss)
        self.history["val_acc"].append(accuracy)

        return avg_loss, accuracy

    def train(self, epochs, checkpoint_dir="outputs/checkpoints"):
        """
        Train student model.

        Args:
            epochs: Number of training epochs
            checkpoint_dir: Directory to save checkpoints
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_acc = 0.0

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = checkpoint_dir / "student_best.pt"
                torch.save(self.student_model.state_dict(), checkpoint_path)
                print(f"  -> Saved best model (Acc: {val_acc:.4f})")

        # Save final model
        final_path = checkpoint_dir / "student_final.pt"
        torch.save(self.student_model.state_dict(), final_path)

        # Save training history
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
