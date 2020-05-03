from modules.functional import multiclass_focal_loss
import torch
import torch.nn.functional as F

class Multiclass_focal_loss(torch.nn.Module):
	def __init__(self, gamma=2.0, alpha=4.0, reduce_type='mean', eps=1e-9):
		"""
		reduce_type (str): 'mean' or 'sum'. Type of reduce.
			default: 'mean'

		eps (float): denominator for numerical stabliity.
			default: 1e-9
		"""
		super(Multiclass_focal_loss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		self.eps = eps
		self.reduce_type = reduce_type

	def reduce(self, fl):
		if self.reduce_type == 'mean':
			return fl.mean()
		elif self.reduce_type == 'max':
			return fl.max()
		else:
			raise ValueError(f'Reduce type {self.reduce_type} not implemented. Should be "max" or "mean".')

	def forward(self, labels, logits):
		fl = multiclass_focal_loss(labels, logits, self.gamma, self.alpha, self.eps)
		return self.reduce(fl)

class Binary_focal_loss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(Binary_focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, targets, inputs):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss