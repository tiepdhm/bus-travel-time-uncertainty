import torch.nn as nn
from bus_benchmark.vendor.convlstm import ConvLSTM


class QuantConvLSTMModel(nn.Module):
    """
    ConvLSTM with Quantile output.
    """

    def __init__(self, output_timesteps: int, num_quantiles: int = 3):
        super().__init__()

        self.output_timesteps = output_timesteps
        self.num_quantiles = num_quantiles

        self.bn0 = PermutedBatchNorm3d(1)

        self.enc0 = ConvLSTM(
            input_dim=1,
            hidden_dim=64,
            kernel_size=(10, 1),
            num_layers=1,
            batch_first=True,
            return_all_layers=False,
        )

        self.do1 = nn.Dropout(0.2)
        self.bn1 = PermutedBatchNorm3d(64)

        self.enc1 = ConvLSTM(
            input_dim=64,
            hidden_dim=64,
            kernel_size=(5, 1),
            num_layers=1,
            batch_first=True,
            return_all_layers=False,
        )

        self.do2 = nn.Dropout(0.1)
        self.bn2 = nn.BatchNorm2d(64)

        self.dec0 = ConvLSTM(
            input_dim=64,
            hidden_dim=64,
            kernel_size=(10, 1),
            num_layers=1,
            batch_first=True,
            return_all_layers=False,
        )

        self.do3 = nn.Dropout(0.1)
        self.bn3 = PermutedBatchNorm3d(64)

        self.dec1 = ConvLSTM(
            input_dim=64,
            hidden_dim=64,
            kernel_size=(5, 1),
            num_layers=1,
            batch_first=True,
            return_all_layers=False,
        )

        # Trả về 1 (mean) + số quantiles, tuân theo thiết kế joint_convLstm
        self.fc = nn.Linear(64, self.num_quantiles + 1)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(
            -1
        )  # (B=batch_size, T=in_steps, H=num_links, W=1, C=1)

        x = x.permute(
            0, 1, 4, 2, 3
        )  # (B=batch_size, T=in_steps, C=1, H=num_links, W=1)

        x = self.bn0(x)

        x, _ = self.enc0(x)
        x = x[0]  # (B=batch_size, T=in_steps, C=64, H=num_links, W=1)

        x = self.do1(x)
        x = self.bn1(x)

        x, _ = self.enc1(x)
        x = x[0]  # (B=batch_size, T=in_steps, C=64, H=num_links, W=1)
        x = x[:, -1]  # (B=batch_size, C=64, H=num_links, W=1)

        x = self.do2(x)
        x = self.bn2(x)

        b, c, h, w = x.shape
        x = x.view(b, -1)  # (B=batch_size, 64*num_links)
        x = x.unsqueeze(1).repeat(
            1, self.output_timesteps, 1
        )  # (B=batch_size, S_out, 64*num_links)
        x = x.view(
            b, self.output_timesteps, c, h, w
        )  # (B=batch_size, S_out, C=64, H=num_links, W=1)

        x, _ = self.dec0(x)
        x = x[0]

        x = self.do3(x)
        x = self.bn3(x)

        x, _ = self.dec1(x)
        x = x[0]  # (B=batch_size, S_out, C=64, H=num_links, W=1)

        x = x.squeeze(-1)  # (B=batch_size, S_out, C=64, H=num_links)
        x = x.permute(0, 1, 3, 2)  # (B=batch_size, S_out, H=num_links, C=64)

        B, S, H, C = x.shape
        x = x.reshape(-1, C)  # (batch_size * S_out * num_links, C=64)
        x = self.fc(x)  # (batch_size * S_out * num_links, C=num_quantiles + 1)
        
        x = x.view(B, S, H, self.num_quantiles + 1)  # (B=batch_size, S_out, H=num_links, num_quantiles + 1)

        return x


class PermutedBatchNorm3d(nn.Module):
    """
    BatchNorm3d with (B, T, C, H, W) shape.
    """

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_features)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.bn(x)
        return x.permute(0, 2, 1, 3, 4)