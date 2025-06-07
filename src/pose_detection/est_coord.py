from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..config import Config
from ..vis import Vis


class EstCoordNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
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

        self.conv4 = nn.Conv1d(1024 + 64, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, 3, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        coord_pred = self.est_coord(pc) # (B, N, 3)

        loss = F.mse_loss(coord_pred, coord)
        metric = dict(
            loss=loss,
            # additional metrics you want to log
        )
        return loss, metric
    
    def est_coord(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Estimate coordinates in the object frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        coord: torch.Tensor
            Estimated coordinates in the object frame, shape \(B, N, 3\)
        """
        
        x = pc.transpose(1, 2) # (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x))) # (B, 64, N)
        point_feat = x # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        global_feat = x.view(x.size(0), -1) # (B, 1024)
        global_feat = global_feat.unsqueeze(2).expand(-1, -1, point_feat.size(2)) # (B, 1024, N)
        x = torch.cat([point_feat, global_feat], dim=1) # (B, 1086, N)
        x = F.relu(self.bn4(self.conv4(x))) # (B, 512, N)
        x = F.relu(self.bn5(self.conv5(x))) # (B, 256, N)
        x = F.relu(self.bn6(self.conv6(x))) # (B, 128, N)
        x = self.conv7(x) # (B, 3, N)
        x = x.transpose(1, 2) # (B, N, 3)
        return x
    
    @torch.no_grad()
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

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        B, N, _ = pc.shape
        dtype, device = pc.dtype, pc.device
        coord = self.est_coord(pc) # (B, N, 3)
        use_ransac = True

        if not use_ransac:
            trans = torch.zeros(B, 3, device=device, dtype=dtype)
            rot = torch.zeros(B, 3, 3, device=device, dtype=dtype)

            for batch_idx in range(B):
                P = pc[batch_idx] # (N, 3)
                Q = coord[batch_idx] # (N, 3)
                P_mean = torch.mean(P, dim=0) # (3,)
                Q_mean = torch.mean(Q, dim=0) # (3,)
                P_centered = P - P_mean # (N, 3)
                Q_centered = Q - Q_mean # (N, 3)

                H = Q_centered.T @ P_centered # (3, 3)
                U, _, Vt = torch.linalg.svd(H)
                det = torch.linalg.det(Vt.T @ U.T)
                Vt[:, -1] *= det.sign()
                R = Vt.T @ U.T # (3, 3)
                rot[batch_idx] = R
                trans[batch_idx] = P_mean - R @ Q_mean
            
            return trans, rot

        # parameter for RANSAC
        iter = 1000
        threshold = 1e-5
        sample_num = 3

        best_rot = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        best_trans = torch.zeros(B, 3, device=device)

        for batch_idx in range(B):
            # P = R @ Q + T
            P = pc[batch_idx] # (N, 3)
            Q = coord[batch_idx]

            max_inlier_cnt = 0

            for _ in range(iter):
                # Randomly sample 3 points
                indices = torch.randperm(N, device=device)[:sample_num]
                P_sample = P[indices] # (3, 3)
                Q_sample = Q[indices] # (3, 3)
                # print(f"debug bs{batch_idx} iter {_}")
                # print(P_sample.shape, P_sample.device, P_sample.dtype)
                # print(Q_sample.shape, Q_sample.device, Q_sample.dtype)

                # Compute the rotation matrix R and translation vector T
                P_mean = torch.mean(P_sample, dim=0)
                Q_mean = torch.mean(Q_sample, dim=0)
                P_centered = P_sample - P_mean
                Q_centered = Q_sample - Q_mean
                # print(P_centered.shape, P_centered.device, P_centered.dtype)
                # print(Q_centered.shape, Q_centered.device, Q_centered.dtype)


                H = Q_centered.T @ P_centered # (3, 3)
                # print("H", H.shape, H.device, H.dtype)

                U, _, Vt = torch.linalg.svd(H)
                det = torch.det(Vt.T @ U.T)
                Vt[:, -1] *= det.sign()
                rot_hyp = Vt.T @ U.T
                trans_hyp = P_mean - rot_hyp @ Q_mean

                # print((rot_hyp @ Q.T).shape)
                # print(trans_hyp.shape)

                P_pred = (rot_hyp @ Q.T + trans_hyp.unsqueeze(1)).T

                loss = torch.sum((P - P_pred) ** 2, dim=1) # (N,)
                inliers = loss < threshold
                inlier_cnt = torch.sum(inliers)

                if inlier_cnt > max_inlier_cnt:
                    max_inlier_cnt = inlier_cnt
                    best_rot[batch_idx] = rot_hyp
                    best_trans[batch_idx] = trans_hyp
                    best_inlier_indices = torch.where(inliers)[0]

            if max_inlier_cnt >= sample_num:
                P_inliers = P[best_inlier_indices]
                Q_inliers = Q[best_inlier_indices]
                P_mean = torch.mean(P_inliers, dim=0)
                Q_mean = torch.mean(Q_inliers, dim=0)
                P_centered = P_inliers - P_mean
                Q_centered = Q_inliers - Q_mean

                H = Q_centered.T @ P_centered
                # print(H.shape)
                U, _, Vt = torch.linalg.svd(H)
                det = torch.det(Vt.T @ U.T)
                Vt[:, -1] *= det.sign()
                best_rot[batch_idx] = Vt.T @ U.T
                best_trans[batch_idx] = P_mean - best_rot[batch_idx] @ Q_mean


        return best_trans, best_rot