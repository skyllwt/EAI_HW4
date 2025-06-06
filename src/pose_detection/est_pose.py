from typing import Tuple, Dict
import torch
from torch import nn

from ..config import Config


class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config

        # PointNet Backbone
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.maxpool = nn.MaxPool1d(kernel_size=1024)

        # Translation Estimation
        self.trans_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

        # Rotation Estimation
        # Use 9-D representation
        self.rotation_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 9),
        )

    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        
        trans_pred, rot_pred = self.est(pc)

        trans_loss = nn.functional.mse_loss(trans_pred, trans)
        rot_loss = nn.functional.mse_loss(rot_pred, rot)

        loss = trans_loss + rot_loss

        metric = dict(
            loss=loss,
            trans_loss=trans_loss,
            rot_loss=rot_loss,
            # additional metrics you want to log
        )
        return loss, metric

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.
        """
        x = pc.transpose(1, 2) # B, 3, N
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))

        x = self.maxpool(x)
        global_feat = x.view(x.size(0), -1) # global feature B, 1024

        trans_pred = self.trans_fc(global_feat) # B, 3
        rot_pred = self.rotation_fc(global_feat) # B, 9

        # Reshape the rotation matrix to 3x3
        rot_pred = rot_pred.view(-1, 3, 3)
        # Apply orthogonalization to the rotation matrix
        
        U, S, Vt = torch.linalg.svd(rot_pred)
        det = torch.det(U @ Vt)
        S = torch.eye(3).unsqueeze(0).repeat(rot_pred.size(0), 1, 1).to(rot_pred.device)
        S[:, 2, 2] = det
        rot_pred = U @ S @ Vt

        return trans_pred, rot_pred
