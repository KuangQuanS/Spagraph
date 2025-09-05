import random
import torch
from torch_geometric.utils import k_hop_subgraph
def augment_graph(edge_index, edge_attr, drop_prob=0.2, perturb_prob=0.1, attr_noise_std=0.1):
    """
    图增强（保持边数不变）
    输入:
        edge_index: [2, E]
        edge_attr: [E, D]
    输出:
        new_edge_index: [2, E]
        new_edge_attr: [E, D]
    """
    E = edge_index.size(1)
    D = edge_attr.size(1)

    # 1. 随机删除一部分边
    keep_mask = torch.rand(E, device=edge_index.device) > drop_prob
    ei_keep = edge_index[:, keep_mask]
    ea_keep = edge_attr[keep_mask]

    # 2. 随机加边（保持总边数不变）
    num_nodes = int(edge_index.max().item()) + 1
    num_add = E - ei_keep.size(1)
    if num_add > 0:
        added_edges = set()
        while len(added_edges) < num_add:
            i = random.randint(0, num_nodes - 1)
            j = random.randint(0, num_nodes - 1)
            if i != j:
                added_edges.add((i, j))
        added_edge_index = torch.tensor(list(added_edges), dtype=torch.long, device=edge_index.device).T
        added_edge_attr = torch.randn(num_add, D, device=edge_attr.device) * attr_noise_std
        ei_new = torch.cat([ei_keep, added_edge_index], dim=1)
        ea_new = torch.cat([ea_keep, added_edge_attr], dim=0)
    else:
        ei_new, ea_new = ei_keep, ea_keep

    # 3. 边扰动概率
    if perturb_prob > 0:
        num_perturb = int(E * perturb_prob)
        for _ in range(num_perturb):
            idx = random.randint(0, E - 1)
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            if src != dst:
                ei_new[:, idx] = torch.tensor([src, dst], device=edge_index.device)

    # 4. 属性加噪
    ea_new += torch.randn_like(ea_new) * attr_noise_std

    # 确保输出形状一致
    assert ei_new.size(1) == E, f"Edge count changed: {ei_new.size(1)} vs {E}"
    assert ea_new.size(0) == E, f"Edge attr count changed: {ea_new.size(0)} vs {E}"

    return ei_new, ea_new

def augment_rw(edge_index, edge_attr, num_nodes, walk_start_ratio=0.2, walk_len=3):
    """
    以随机起始点进行多次随机游走，并取k-hop子图
    """
    device = edge_index.device

    num_walks = int(num_nodes * walk_start_ratio)
    start_nodes = torch.randperm(num_nodes)[:num_walks].tolist()

    # 收集访问到的节点
    visited = set(start_nodes)
    adj = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adj[u].append(v)
        adj[v].append(u)  # 无向图

    for start in start_nodes:
        curr = start
        for _ in range(walk_len):
            neighbors = adj[curr]
            if len(neighbors) == 0:
                break
            curr = random.choice(neighbors)
            visited.add(curr)

    # 子图节点索引
    visited = list(visited)
    visited_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    visited_mask[visited] = True

    # 从原始图中提取子图
    node_idx, new_edge_index, mapping, edge_mask = k_hop_subgraph(
        visited, num_hops=1, edge_index=edge_index, relabel_nodes=True
    )
    new_edge_attr = edge_attr[edge_mask]

    return new_edge_index, new_edge_attr
