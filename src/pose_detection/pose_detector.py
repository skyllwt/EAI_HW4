from ..config import Config
from .est_pose import EstPoseNet
from .est_coord import EstCoordNet
from ..utils import to_pose
import torch

config = Config()
model = EstCoordNet(config)

ckpt = torch.load("src/pose_detection/model_ckpt.pth", map_location="cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(ckpt["model"])
model = model.eval().to(device)

def detect_pose(point_cloud):
    """
    Detect pose in camera frame from point cloud using the EstPoseNet model.
    """
    point_cloud = torch.tensor(point_cloud, device=device, dtype=torch.float32)
    with torch.no_grad():
        trans_pred, rot_pred = model.est(point_cloud.reshape(1, -1, 3))
    return to_pose(trans_pred.cpu().numpy(), rot_pred.cpu().numpy())
