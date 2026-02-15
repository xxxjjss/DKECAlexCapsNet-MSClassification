import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class PrimaryCaps(nn.Module):
    def __init__(self, num_caps=32, in_channel=256, out_channel=8, kersel_size=9, stride=2, padding=0):
        super(PrimaryCaps, self).__init__()
        self.num_caps = num_caps
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kersel_size,
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
        u = u.view(batch_size, 8, -1)
        u = u.transpose(1, 2)
        u_squashed = self.squash(u)
        return u_squashed

    def squash(self, u):
        batch_size = u.size(0)
        square = u ** 2
        square_sum = torch.sum(square, dim = 2)
        norm = torch.sqrt(square_sum)
        factor = norm**2 / (norm * (1 + norm**2))
        u_squashed = factor.unsqueeze(2)
        u_squashed = u_squashed * u
        return u_squashed

class DenseCapsule(nn.Module):
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, device, routings=3):
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
        x_hat_detached = x_hat
        b = torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps).to(self.device)
        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            c = F.softmax(b, dim=1)
            if i == self.routings - 1:
                outputs = self.squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
            else:
                outputs = self.squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)
        return torch.squeeze(outputs, dim=-2)
    def squash(self, inputs, axis=-1):
        norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
        scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
        return scale * inputs