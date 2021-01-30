from spektral.data import Graph
from spektral.datasets import QM9

dataset = QM9(amount=1000)  # Set amount=None to train on whole dataset

g9=dataset[0]
print('g9 {}\n\n{}'.format(type(g9.a),g9.y))

g=Graph()
g.a=[]