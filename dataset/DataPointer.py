import torch

class DataPointer(object):
	"""docstring for DataPointer"""
	def __init__(self, n_total):
		super(DataPointer, self).__init__()
		self.n_total = n_total
		if self.n_total > 0:
			self.idx_perm = torch.randperm(self.n_total)
			self.current_pos = 0
		else:
			self.idx_perm = None
			self.current_pos = None

	def load_next_batch(self,batch_size):
		if not self.n_total > 0:
			return None

		if batch_size == 0:
			return None

		# get indices
		if batch_size + self.current_pos <= self.n_total:
			indices = self.idx_perm.narrow(0, self.current_pos, batch_size)
		else:
			rest = batch_size + self.current_pos - self.n_total
			part1 = self.idx_perm.narrow(0, self.current_pos, self.n_total-self.current_pos)
			part2 = self.idx_perm.narrow(0, 0, rest)
			indices = torch.cat([part1, part2])

		# update pointer
		self.current_pos += batch_size
		if self.current_pos >= self.n_total:
			self.current_pos = 0
			self.idx_perm = torch.randperm(self.n_total)

		# print(indices)
		return indices
		# return torch.Tensor([1,1,1,1,1,1])

if __name__ == '__main__':
	d = DataPointer(10)
	for i in range(0,10):
		indices = d.load_next_batch(3)
		print(indices)