import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """

    [Conv -> BN -> ReLu] * 2

    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class SiamInconv(nn.Module):
	def __init__(self, in_channels=1, out_channels=32):
		super(SiamInconv, self).__init__()

		self.conv_1 = DoubleConv(in_channels, out_channels)
		self.conv_2 = DoubleConv(out_channels, out_channels)
		self.conv_3 = DoubleConv(out_channels, out_channels)
		self.conv_4 = DoubleConv(out_channels, out_channels)
		self.conv_5 = DoubleConv(out_channels, out_channels)

	def forward(self, x):
		x = self.conv_1(x)
		x = self.conv_2(x) + x
		x = self.conv_3(x) + x
		x = self.conv_4(x) + x
		x = self.conv_5(x) + x
		return x

class SCNet(nn.Module):
	def __init__(self, input_size=3, num_inputs=5, in_channels=1, out_channels=32, num_classes=7):
		super(SCNet, self).__init__()

		self.input_size = input_size
		self.out_channels = out_channels

		self.siam_inputs = []
		for i in range(num_inputs):
			self.__setattr__(f'siam_head_{i}', SiamInconv(in_channels=in_channels, out_channels=out_channels))
			self.siam_inputs.append(
				self.__getattr__(f'siam_head_{i}')
				)

		self.ConvBlock = nn.Sequential(
			DoubleConv(in_channels=out_channels, out_channels=out_channels),
			DoubleConv(in_channels=out_channels, out_channels=out_channels),
			DoubleConv(in_channels=out_channels, out_channels=out_channels)
			)

		self.LinearBlock = nn.Sequential(
			nn.Linear((input_size ** 2) * out_channels, 256),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(256),
			nn.Linear(256, 256),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(256),
			nn.Linear(256, num_classes)			
			)

	def forward(self, x):

		for i in range(x.shape[1]):
			if i == 0:
				x_in = self.siam_inputs[i](x[:, i, :, :, :])
			else:
				x_in += self.siam_inputs[i](x[:, i, :, :, :])

		x = self.ConvBlock(x_in)
		x = self.LinearBlock(x.view(-1, (self.input_size ** 2) * self.out_channels)	)
		return x			


