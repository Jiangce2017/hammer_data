#### Data conversion
import abc
import numpy as np
import numpy.linalg as LA

class PhysicalDataInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_data_class') and 
                callable(subclass.get_data_class) and 
                hasattr(subclass, 'load_data_source') and 
                callable(subclass.load_data_source) and 
                hasattr(subclass, 'to') and 
                callable(subclass.to) or 
                NotImplemented)
                
    @abc.abstractmethod
    def load_data_source(self,path:str):
        """Load in the data set"""
        raise NotImplementedError
        
    @abc.abstractmethod
    def get_data_class(self):
        """Get the data type"""
        raise NotImplementedError
        
    @abc.abstractmethod
    def to(self, data_class:str):
        """Convert the data to the data_class"""
        raise NotImplementedError

        
class MeshDataClass(PhysicalDataInterface):
    def __init__(self,mesh_data=None):
        self.data_class = "Mesh"
        self.points = None 
        self.cells = None
        self.point_values = None
        self.cell_values = None
        if mesh_data != None:
            self.set_data(mesh_data)

    def load_data_source(self, path:str):
        pass
    
    def get_data_class(self):
        return self.data_class
        
    def set_data(self,mesh_data):
        self.points = mesh_data["points"]
        self.cells = mesh_data["cells"]
        self.point_values = mesh_data["point_values"]
        self.cell_values = mesh_data["cell_values"]
    
    def to(self,data_class:str)-> object:
        if data_class == "Graph":
            obj = GraphDataClass()
        elif data_class == "Voxel":
            obj = VoxelDataClass()
        else:
            raise NotImplementedError
        data = self.to_data(data_class)
        obj.set_data(data)
        return obj
    
    def to_data(self,data_class:str):
        if data_class == "Graph":
            data = self.mesh2graph()
                
        elif data_class == "Voxel":
            data = self.mesh2voxel()
        else:
            raise NotImplementedError
        return data
        
    def mesh2graph(self):
        voxel_data = self.mesh2voxel()
        voxel = VoxelDataClass(voxel_data)
        graph_data = voxel.to_data("Graph")
        return graph_data
        
    def mesh2voxel(self):
        center_coords = np.mean(self.points[self.cells],axis=1)
        dx = LA.norm(self.points[self.cells[0,0]] - self.points[self.cells[0,1]])
        int_coords = self._coord2inds(center_coords,dx)
        voxel_data = {"center_coords":center_coords,
                      "int_coords":int_coords,
                      "voxel_values": self.cell_values,
                      "dx": dx}
        return voxel_data
        
    def _coord2inds(self, float_coords, dx):
        float_min = np.min(float_coords,axis=0,keepdims=True)
        int_coords = np.round((float_coords-float_min)/dx).astype(int)
        return int_coords
        


class VoxelDataClass(PhysicalDataInterface):
    def __init__(self,voxel_data=None):
        self.data_class = "Voxel"
        self.center_coords = None
        self.int_coords = None
        self.voxel_values = None
        self.dx = None
        if voxel_data != None:
            self.set_data(voxel_data)

    def load_data_source(self, path:str):
        pass
    
    def get_data_class(self):
        return self.data_class
        
    def set_data(self,voxel_data):
        self.center_coords = voxel_data["center_coords"]
        self.int_coords = voxel_data["int_coords"]
        self.voxel_values = voxel_data["voxel_values"]
        self.dx = voxel_data["dx"]
    
    def to(self,data_class:str)-> object:
        if data_class == "Graph":
            obj = GraphDataClass()
        elif data_class == "Mesh":
            obj = MeshDataClass()
        else:
            raise NotImplementedError
        data = self.to_data(data_class)
        obj.set_data(data)
        return obj
    
    def to_data(self,data_class:str)-> object:
        if data_class == "Graph":
            data = self.voxel2graph()
                
        elif data_class == "Mesh":
            data = self.voxel2mesh()
        else:
            raise NotImplementedError
        return data
        
    def voxel2graph(self):
        int_coords = self.int_coords
        max_inds = np.max(int_coords, axis=0)
        inds_hash = self._grid_hash(int_coords, max_inds)
        lookup_table = self._create_lookup_table(inds_hash, max_inds)

        origins = np.arange(int_coords.shape[0], dtype=np.int32)
        moves = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.int32)
        edges = []
        for i_move in range(3):
            moved_inds = int_coords + moves[i_move]
            mask_in = (moved_inds[:, 0] >= 0) & (moved_inds[:, 0] <= max_inds[0]) & (moved_inds[:, 1] >= 0) & (
                        moved_inds[:, 1] <= max_inds[1]) & \
                      (moved_inds[:, 2] >= 0) & (moved_inds[:, 2] <= max_inds[2])
            moved_inds_hash = self._grid_hash(moved_inds[mask_in], max_inds)
            end0 = origins[mask_in]
            end1 = lookup_table[moved_inds_hash]
            mask_act = end1 >= 0
            edg = np.stack([end0[mask_act], end1[mask_act]], axis=0)
            edges.append(edg)
        edges_total = np.concatenate(edges, axis=1)
        node_features = np.concatenate((self.center_coords, self.voxel_values), axis=1)
        
        graph_data = {"node_features": node_features,
                      "edge_index": edges_total}
        return graph_data
                      


    def _grid_hash(self, arr, max_inds):
        int_hash = arr[:, 0] + arr[:, 1] * (max_inds[0] + 1) + arr[:, 2] * (max_inds[0] + 1) * (max_inds[1] + 1)
        return int_hash

    def _create_lookup_table(self, arr_hash, max_inds):
        lookup_table = -1 * np.ones(((max_inds[0] + 1) * (max_inds[1] + 1) * (max_inds[2] + 1)), dtype=np.int64)
        lookup_table[arr_hash] = np.arange(arr_hash.shape[0])
        return lookup_table
        
    def voxel2mesh(self):
        pass
        
        
class GraphDataClass(PhysicalDataInterface):
    def __init__(self,graph_data=None):
        self.data_class = "Graph"
        self.node_features = None
        self.edge_index = None
        if graph_data != None:
            self.set_data(graph_data)

    def load_data_source(self, path:str):
        pass
    
    def get_data_class(self):
        return self.data_class
    
    def set_data(self,graph_data):
        self.node_features = graph_data["node_features"]
        self.edge_index = graph_data["edge_index"]
    
    def to(self,data_class:str)-> object:
        if data_class == "Voxel":
            obj = VoxelDataClass()
        elif data_class == "Mesh":
            obj = MeshDataClass()
        else:
            raise NotImplementedError
        data = self.to_data(data_class)
        obj.set_data(data)
        return obj
    
    def to_data(self,data_class:str)-> object:
        if data_class == "Mesh":
                data = graph2mesh(self)
                
        if data_class == "Voxel":
            data = graph2voxel(self)
        else:
            raise NotImplementedError
        return data
        
    def graph2mesh(self):
        pass
        
    def graph2voxel(self):
        pass
        
    
        
    
        