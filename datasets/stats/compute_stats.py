import math
import os
import sys

import numpy as np
import scipy.io

import torch
from tqdm.auto import tqdm


def get_stats(X):
	X = X.double()
	mu = X.mean(dim=0)
	X = X - mu
	# X = U diag(S) VT
	# cov = XT X / (n - 1)
	#     = V diag(S^2 / (n - 1)) VT
	#
	# NB cov(X W) = WT cov(X) W.
	#
	# So the linear proj to have idt cov should be
	# W = V diag( sqrt(n - 1) / S)
	#
	# W^-1 = diag( S / sqrt(n - 1) ) VT
	U, s, V = X.svd(some=True)
	n = X.shape[0]
	W = V @ (math.sqrt(n - 1) / s).diag()
	Winv = (s / math.sqrt(n - 1)).diag() @ V.T
	return dict(
		c=np.cov(X.T.numpy()),
		w=W.numpy(),
		winv=Winv.numpy(),
		mu=mu.numpy()
	)


if __name__ == '__main__':
	X = torch.empty(10000, 128, 128, 3).permute(0, 3, 1, 2)

	for i in tqdm(range(X.shape[0])):
		X[i] = torch.load(os.path.join(sys.argv[1], f"{i:06d}.pth"), map_location='cpu')

	X = X.permute(0, 2, 3, 1).flatten(0, 2)

	stats = get_stats(X)

	scipy.io.savemat(os.path.join(sys.argv[1], 'stats.mat'), stats)

