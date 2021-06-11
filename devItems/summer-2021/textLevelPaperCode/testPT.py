import dgl
import torch as th
u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])
g = dgl.graph((u, v))
g.ndata['x'] = th.randn(5, 3)  # original feature is on CPU
g.device
cuda_g = g.to('cuda:0')  # accepts any device objects from backend framework
cuda_g.device
cuda_g.ndata['x'].device       # feature data is copied to GPU too

# A graph constructed from GPU tensors is also on GPU
u, v = u.to('cuda:0'), v.to('cuda:0')
g = dgl.graph((u, v))
g.device
