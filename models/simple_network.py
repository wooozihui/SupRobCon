import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
	def __init__(self):
		super(Head,self).__init__()
	
	def forward(self,x):
		x = x- 0.5
		x = x*2
		return x
				 

class Mymodel(nn.Module):
	def __init__(self):
		super(Mymodel, self).__init__()
		self.body = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=1,padding=1, bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True),
			nn.Conv2d(16, 32, kernel_size=3, stride=1,padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 16, kernel_size=3, stride=1,padding=1, bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True),
			nn.Conv2d(16, 8, kernel_size=3, stride=1,padding=1, bias=False),
			nn.BatchNorm2d(8),
			nn.ReLU(inplace=True),
			
			)
		self.fc = nn.Linear(8*32*32, 10)
	def forward(self,input):
		batchsize = input.size()[0]
		out = self.body(input)
		out = out.view(batchsize,-1)
		out = self.fc(out)
		return out

