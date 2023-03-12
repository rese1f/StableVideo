import torch.nn as nn

class AGGNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1,bias=False),
            nn.ReLU()
        )
        self.stage2=nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=3,padding=1,bias=False),
        )

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        return x + x2