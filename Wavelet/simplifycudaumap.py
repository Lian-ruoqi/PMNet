import torch 
from torch import nn 
import torch.nn.functional as F
import faiss
import numpy as np

def compute_knn(
    X: torch.Tensor,
    k: int,
):

    X = X.float()
    Xh = X.half()                                 
    dist_h = torch.cdist(Xh, Xh, p=2)             
    _, nbrs_k1 = dist_h.topk(k + 1,                
                            largest=False,
                            dim=1)              
    neighbors = nbrs_k1[:, 1:]                 
    diff = X.unsqueeze(1) - X[neighbors]         
    distances = torch.norm(diff, p=2, dim=2)      

    return distances, neighbors

def compute_edge_probs(
    distances: torch.Tensor, 
    neighbors: torch.Tensor, 
    sigma: torch.Tensor,      
):

    device = distances.device
    N, k = distances.shape
    rho = distances[:, 0]                                                
    diff = torch.clamp(distances - rho.unsqueeze(1), min=0.0)     
    p_ij = torch.exp(- diff / sigma)                                    
    src = torch.repeat_interleave(torch.arange(N, device=device), k)      
    dst = neighbors.reshape(-1)                                            
    vals = p_ij.reshape(-1)                                                
    mask = dst >= 0
    src = src[mask];   dst = dst[mask];   vals = vals[mask]

    return src, dst, vals

def symmetrize_p(pos_src: torch.Tensor,
                 pos_dst: torch.Tensor,
                 pos_vals: torch.Tensor,
                 N: int,
                 eps: float):
   
    src = torch.cat([pos_src, pos_dst], dim=0)
    dst = torch.cat([pos_dst, pos_src], dim=0)
    vals = torch.cat([pos_vals, pos_vals], dim=0)
    u = torch.minimum(src, dst)
    v = torch.maximum(src, dst)
    keys = u * N + v  # [2E]
    unique_keys, inv_idx, counts = torch.unique(
        keys, return_inverse=True, return_counts=True
    )
    sum_p = torch.zeros_like(unique_keys, dtype=vals.dtype, device=vals.device)
    sum_p = sum_p.index_add(0, inv_idx, vals) 
    log_vals = torch.log(vals + eps)
    sum_log = torch.zeros_like(unique_keys, dtype=log_vals.dtype, device=vals.device)
    sum_log = sum_log.index_add(0, inv_idx, log_vals)
    prod_p = torch.exp(sum_log)
    mask2 = (counts == 2).to(vals.dtype)
    p_sym = sum_p - prod_p * mask2
    pos_src = (unique_keys // N).long()
    pos_dst = (unique_keys %  N).long()
    pos_vals = p_sym.clamp(eps, 1.0 - eps)

    return pos_src, pos_dst, pos_vals    

class ParametricUMAPModule(nn.Module):
   
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k: int = 8,
        num_negatives: int = 1,
        a: float = 1.0,
        b: float = 1.0,
        eps: float = 1e-8,
        factor: int = 2,    
    ):
        super().__init__()
      
        self.k = k
        self.num_negatives = num_negatives
        self.a, self.b, self.eps, self.factor = a, b, eps, factor
        self.mlp = nn.Linear(in_dim, out_dim)
        self.sigma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
       
        x = x.float()
        Z = self.mlp(x)  
        if not self.training:
            return Z, None
        with torch.no_grad():
            distances, neighbors = compute_knn(x, self.k)  
        distances = distances.detach()
        neighbors = neighbors.detach()
        sigma = self.sigma.clamp(min=self.eps)
        pos_src, pos_dst, pos_vals = compute_edge_probs(distances, neighbors, sigma)
        N = x.size(0)
        pos_src, pos_dst, pos_vals = symmetrize_p(pos_src, pos_dst, pos_vals, N, self.eps)
        E = pos_src.numel()  
        total_neg = self.num_negatives * E               
        cand = self.factor * total_neg                   
        i = torch.randint(0, N, (cand,), device=x.device)
        j = torch.randint(0, N, (cand,), device=x.device)
        mask = (i != j)                                
        pos_pairs = pos_src * N + pos_dst                 
        cand_pairs = i * N + j
        mask &= ~torch.isin(cand_pairs, pos_pairs)       
        i = i[mask][:total_neg]                          
        j = j[mask][:total_neg]
        src_all = torch.cat([pos_src, i], dim=0)
        dst_all = torch.cat([pos_dst, j], dim=0)
        labels_pos = pos_vals                        
        labels_neg = torch.zeros(i.size(0), device=x.device, dtype=labels_pos.dtype)
        labels = torch.cat([labels_pos, labels_neg], dim=0)
        z_src = Z[src_all]        
        z_dst = Z[dst_all]
        z_dist = torch.norm(z_src - z_dst, dim=1)
        q = (1 + self.a * z_dist.pow(2 * self.b)).pow(-1)
        q = q.clamp(self.eps, 1.0 - self.eps)
        loss = F.binary_cross_entropy_with_logits(q, labels)

        return Z, loss
    