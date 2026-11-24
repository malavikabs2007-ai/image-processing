"""A small PointPillars-style pillar feature baseline (not a trained detector)."""
import numpy as np

def make_pillars(points, voxel_size=(0.25,0.25), x_range=(0,70.4), y_range=(-40,40), max_points=32, max_pillars=12000):
    """Quantize returns into vertical pillars and return padded point features + indices."""
    sx,sy=voxel_size; xyz=points[:,:3]
    ix=np.floor((xyz[:,0]-x_range[0])/sx).astype(np.int32); iy=np.floor((xyz[:,1]-y_range[0])/sy).astype(np.int32)
    nx=int(np.ceil((x_range[1]-x_range[0])/sx)); ny=int(np.ceil((y_range[1]-y_range[0])/sy))
    valid=(ix>=0)&(ix<nx)&(iy>=0)&(iy<ny); valid_ids=np.flatnonzero(valid); groups={}
    for index,(x,y) in enumerate(zip(ix[valid],iy[valid])):
        point_index=int(valid_ids[index]); key=(int(x),int(y))
        if key not in groups and len(groups)>=max_pillars: continue
        bucket=groups.setdefault(key,[])
        if len(bucket)<max_points: bucket.append(point_index)
    keys=list(groups); features=np.zeros((len(keys),max_points,4),np.float32); counts=np.zeros(len(keys),np.int32)
    for row,key in enumerate(keys):
        ids=groups[key]; vals=points[ids,:4]; counts[row]=len(ids)
        # Features are XYZ offsets from the pillar mean plus intensity.
        features[row,:len(ids),:3]=vals[:,:3]-vals[:,:3].mean(axis=0); features[row,:len(ids),3]=vals[:,3]
    return np.asarray(keys,np.int32),features,counts

def build_torch_pillar_encoder():
    """Create a learnable Torch pillar encoder; weights and training are not bundled."""
    try: import torch
    except ImportError as exc: raise RuntimeError("Install a PyTorch build before using the optional pillar encoder") from exc
    class PillarEncoder(torch.nn.Module):
        """Pool learned point features into one feature vector per pillar."""
        def __init__(self):
            super().__init__(); self.point_mlp=torch.nn.Sequential(torch.nn.Linear(4,32),torch.nn.ReLU(),torch.nn.Linear(32,64),torch.nn.ReLU())
        def forward(self,features,mask):
            encoded=self.point_mlp(features); encoded=encoded.masked_fill(~mask.unsqueeze(-1),-1e4)
            return encoded.max(dim=1).values
    return PillarEncoder()
