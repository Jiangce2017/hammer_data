#### test am_data_class
import pickle
import meshio
import numpy as np
import numpy.linalg as LA
import torch
from hammer_data import PhysicalDataInterface, MeshDataClass, GraphDataClass, VoxelDataClass
from am_data_class import AMMesh, AMVoxel, AMGraph

mesh = AMMesh()
path = "./data/T0009910.vtu"
mesh.load_data_source(path)
voxel = mesh.to("Voxel")
print(voxel)

graph = mesh.to("Graph")
node_features, edge_index = graph.torch_data()
print(node_features)
print(edge_index)

graph.plot_graph()