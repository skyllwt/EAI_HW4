from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn

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
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp1_res = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1),
        )
        # raise NotImplementedError("You need to implement some modules here")

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
        # raise NotImplementedError("You need to implement the forward function")
        
        pc = pc.transpose(1, 2)
        pc_after_1 = self.mlp1(pc)
        pc_after_1_res = self.mlp1_res(pc_after_1)
        pc_after_1_res = torch.amax(pc_after_1_res, dim=2).unsqueeze(2).expand(-1, -1, pc_after_1.shape[2])
        pc_full = torch.cat([pc_after_1, pc_after_1_res], dim=1)
        pred = self.mlp2(pc_full)
        pred = pred.transpose(1, 2)
        loss = nn.MSELoss()(pred, coord)
        
        metric = dict(
            loss=loss,
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

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        batch_size, num_points, _ = pc.shape
        device, dtype = pc.device, pc.dtype
        
        pc_trans = pc.transpose(1, 2)
        pc_after_1 = self.mlp1(pc_trans)
        pc_after_1_res = self.mlp1_res(pc_after_1)
        pc_after_1_res = torch.amax(pc_after_1_res, dim=2).unsqueeze(2).expand(-1, -1, pc_after_1.shape[2])
        pc_full = torch.cat([pc_after_1, pc_after_1_res], dim=1)
        pred_coords = self.mlp2(pc_full).transpose(1, 2)
        
        final_rot = torch.eye(3, device=device, dtype=dtype).repeat(batch_size, 1, 1)
        final_trans = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        
        for bidx in range(batch_size):
            src = pc[bidx].detach().cpu().numpy()
            dst = pred_coords[bidx].detach().cpu().numpy()
            
            best_rot = np.eye(3)
            best_trans = np.zeros(3)
            max_inliers = 0
            inlier_mask = None
            
            for _ in range(1000):
                for attempt in range(10):
                    sample_idx = np.random.choice(num_points, 3, replace=False)
                    sample_src = src[sample_idx]
                    sample_dst = dst[sample_idx]

                    vec1 = sample_src[1] - sample_src[0]
                    vec2 = sample_src[2] - sample_src[0]
                    if np.linalg.norm(np.cross(vec1, vec2)) > 1e-7:
                        break
                else:
                    continue
                
                try:
                    R, t = self._calc_transformation(sample_dst, sample_src)
                except np.linalg.LinAlgError:
                    continue
                
                error = np.linalg.norm((R @ dst.T).T + t - src, axis=1)
                curr_inliers = np.sum(error < 0.005)
                
                if curr_inliers > max_inliers:
                    best_rot = R
                    best_trans = t
                    max_inliers = curr_inliers
                    inlier_mask = error < 0.005
            
            if max_inliers >= 3:
                try:
                    R_refined, t_refined = self._calc_transformation(
                        dst[inlier_mask], src[inlier_mask]
                    )
                    final_rot[bidx] = torch.tensor(R_refined, device=device, dtype=dtype)
                    final_trans[bidx] = torch.tensor(t_refined, device=device, dtype=dtype)
                except np.linalg.LinAlgError:
                    pass
            else:
                final_rot[bidx] = torch.tensor(best_rot, device=device, dtype=dtype)
                final_trans[bidx] = torch.tensor(best_trans, device=device, dtype=dtype)
        
        return final_trans, final_rot

    def _calc_transformation(self, q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p_centroid = p.mean(0)
        q_centroid = q.mean(0)
        p_centered = p - p_centroid
        q_centered = q - q_centroid
        
        cov_mat = q_centered.T @ p_centered + 1e-8 * np.eye(3)
        U, _, Vt = np.linalg.svd(cov_mat)
        
        det = np.linalg.det(Vt.T @ U.T)
        reflect = np.diag([1, 1, -1]) if det < 0 else np.eye(3)
        
        R = Vt.T @ reflect @ U.T
        t = p_centroid - R @ q_centroid
        
        U_ortho, _, Vt_ortho = np.linalg.svd(R)
        return U_ortho @ Vt_ortho, t
