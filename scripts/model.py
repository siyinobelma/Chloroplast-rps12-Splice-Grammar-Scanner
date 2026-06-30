import torch.nn as nn
import torch

class SplicingCNN(nn.Module):
    """三通道CNN，分别处理外显子1、内含子、外显子2，然后融合"""
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv1d(4, config.NUM_FILTERS, config.KERNEL_SIZE, padding='same')
        self.conv2 = nn.Conv1d(config.NUM_FILTERS, config.NUM_FILTERS, config.KERNEL_SIZE, padding='same')
        self.conv3 = nn.Conv1d(config.NUM_FILTERS, config.NUM_FILTERS, config.KERNEL_SIZE, padding='same')
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(config.DROPOUT)
        # 三个通道特征拼接后接全连接
        self.fc1 = nn.Linear(config.NUM_FILTERS * 3, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # x: (batch, 3, 4, L)  3通道：e1, intron, e2
        e1 = x[:, 0]   # (B,4,L)
        intr = x[:, 1]
        e2 = x[:, 2]
        # 共享卷积权重或独立？这里独立处理，也可共享
        e1 = self.pool(torch.relu(self.conv3(torch.relu(self.conv2(torch.relu(self.conv1(e1))))))).squeeze(-1)
        intr = self.pool(torch.relu(self.conv3(torch.relu(self.conv2(torch.relu(self.conv1(intr))))))).squeeze(-1)
        e2 = self.pool(torch.relu(self.conv3(torch.relu(self.conv2(torch.relu(self.conv1(e2))))))).squeeze(-1)
        concat = torch.cat([e1, intr, e2], dim=1)
        concat = self.dropout(concat)
        out = torch.relu(self.fc1(concat))
        out = self.fc2(out)
        return out