import torch
import torch.nn as nn
import torch.nn.functional as F
class AlexCapsNet(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2),
            DK_ECA(32, base_k=7),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(3, stride=2),
            DK_ECA(64, base_k=5),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(3, stride=2),
            DK_ECA(64, base_k=3)
        )
        self.Cap = nn.Sequential(
            PrimaryCaps(
                num_caps=32,
                in_channel=64,
                out_channel=8,
                kernel_size=5,
                stride=2,
                padding=0
            ),
            DenseCapsule(
                in_num_caps=4928,
                in_dim_caps=8,
                out_num_caps=4,
                out_dim_caps=32,
                device=device,
                routings=5
            )
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)

        return x.norm(dim=-1)
