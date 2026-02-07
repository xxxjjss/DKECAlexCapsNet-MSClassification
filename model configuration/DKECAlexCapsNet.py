import torch
import torch.nn as nn
import torch.nn.functional as F
class PrimaryCaps(nn.Module):
    def __init__(self, num_caps, in_channel, out_channel, kernel_size, stride, padding):
        super(PrimaryCaps, self).__init__()
        self.num_caps = num_caps
        self.capsules = nn.ModuleList([
            nn.Conv1d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding)
            for i in range(num_caps)
        ])
    def forward(self, x):
        batch_size = x.size(0)
        u = []
        for i in range(self.num_caps):
            u_i = self.capsules[i](x)
            u_i = u_i.view(batch_size, 8, -1, 1)
            u.append(u_i)
        u = torch.cat(u, dim=3)
        u_squashed = self.squash(u)
        return u_squashed
    def squash(self, u):
        batch_size = u.size(0)
        square = u ** 2
        square_sum = torch.sum(square, dim=2)
        norm = torch.sqrt(square_sum)
        factor = norm ** 2 / (norm * (1 + norm ** 2))
        u_squashed = factor.unsqueeze(2)
        u_squashed = u_squashed * u
        return u_squashed
class DenseCapsule(nn.Module):
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, device, routings):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.device = device
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))
    def forward(self, x):
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)
        b = torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps).to(self.device)
        return torch.squeeze()
    def squash(self, inputs, axis=-1):
        norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
        norm_safe = norm + 1e-8
        scale = norm_safe ** 2 / (1 + norm_safe ** 2) / norm_safe
        return scale * inputs
class DK_ECA(nn.Module):
    def __init__(self, channel, base_k):
        super(DK_ECA, self).__init__()
        self.base_k = base_k | 1
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.k_gen = nn.Sequential(
            nn.Linear(channel, max(4, channel // 8)),
            nn.GELU(),
            nn.Linear(max(4, channel // 8), 1),
            nn.Tanh()
        )
        nn.init.constant_(self.k_gen[2].bias, 0.5)
        nn.init.normal_(self.k_gen[2].weight, std=0.01)
    def forward(self, x):
        b, c, l = x.shape
        gap = self.gap(x).view(b, c)
        delta_k = self.k_gen(gap) * 2
        delta_k = delta_k.round().clamp(-3, 3).long()
        k = (self.base_k + delta_k) | 1
        max_k = 9
        y = self.gap(x).transpose(1, 2)
        padded_y = F.pad(y, ((max_k - 1) // 2, (max_k - 1) // 2))
        weights = []
        for bi in range(b):
            k_size = k[bi].item()
            weight = torch.zeros(max_k, device=x.device)
            start = (max_k - k_size) // 2
            weight[start:start + k_size] = 1.0 / k_size
            weights.append(weight.view(1, 1, -1))
        weights = torch.cat(weights, dim=0)
        y = F.conv1d(
            input=padded_y.view(1, b, -1),
            weight=weights,
            groups=b,
            padding=0
        ).view(b, 1, -1)
        sigmoid_y = torch.sigmoid(y.transpose(1, 2))
        return x * sigmoid_y
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