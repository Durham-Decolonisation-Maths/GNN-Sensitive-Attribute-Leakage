import os
import urllib.request
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

def download_and_build_pokec_z(
    output_path="pokec_z_graph.pt", 
    max_nodes=2000, 
    sample_subgraph=True
):
    """
    Downloads raw Pokec-z data from the open-source FairGNN benchmark repository,
    processes node features, target labels, and graph topology, and saves a 
    PyTorch Geometric Data object.
    
    Args:
        output_path (str): Destination path for the saved .pt file.
        max_nodes (int): Sub-sample node limit for fast CPU execution in workshops.
        sample_subgraph (bool): If True, crops the graph to `max_nodes` while preserving edges.
    """
    print("=== Downloading Pokec-z Dataset ===")
    
    # Raw data sources from canonical FairGNN / PyGDebias open repositories
    base_url = "https://raw.githubusercontent.com/yushundong/PyGDebias/main/src/pygdebias/datasets/pokec/"
    csv_url = base_url + "region_job.csv"
    relation_url = base_url + "region_job_relationship.txt"
    
    os.makedirs("./temp_pokec", exist_ok=True)
    csv_path = "./temp_pokec/region_job.csv"
    rel_path = "./temp_pokec/region_job_relationship.txt"
    
    if not os.path.exists(csv_path):
        print("Fetching node attributes (region_job.csv)...")
        urllib.request.urlretrieve(csv_url, csv_path)
        
    if not os.path.exists(rel_path):
        print("Fetching graph edges (region_job_relationship.txt)...")
        urllib.request.urlretrieve(relation_url, rel_path)
    df = pd.read_csv(csv_path)
    
    # Extract Sensitive Attribute (S): Region (1 = Žilina region, 0 = Other)
    sens_attr = df["region"].values.astype(np.float32)
    
    # Extract Target Label (Y): Working field classification
    labels = df["I_am_working_in_field"].values.astype(np.int64)
    
    # Extract Feature Matrix (X): Remove ID, region, and target column
    feature_cols = [c for c in df.columns if c not in ["user_id", "region", "I_am_working_in_field"]]
    features = df[feature_cols].values.astype(np.float32)
    
    # 2. Load Relationship Edges
    edges_df = pd.read_csv(rel_path, sep="\t", header=None, names=["src", "dst"])
    
    # Map raw user IDs to zero-indexed node indices
    user_id_map = {uid: idx for idx, uid in enumerate(df["user_id"].values)}
    
    # Filter edges to ensure both endpoints exist in the feature set
    valid_mask = edges_df["src"].isin(user_id_map) & edges_df["dst"].isin(user_id_map)
    edges_df = edges_df[valid_mask]
    
    src_indices = edges_df["src"].map(user_id_map).values
    dst_indices = edges_df["dst"].map(user_id_map).values
    
    edge_index = torch.tensor(np.vstack((src_indices, dst_indices)), dtype=torch.long)
    
    # 3.Sub-sampling to extract snippet of data 
    if sample_subgraph and len(df) > max_nodes:
        print(f"Sub-sampling graph to top {max_nodes} connected nodes for quick CPU runtime...")
        node_subset = torch.arange(max_nodes)
        
        # Keep features, labels, and sensitive attributes for subset
        features = features[:max_nodes]
        labels = labels[:max_nodes]
        sens_attr = sens_attr[:max_nodes]
        
        # Keep edges where both source and target are within the sub-graph
        edge_mask = (edge_index[0] < max_nodes) & (edge_index[1] < max_nodes)
        edge_index = edge_index[:, edge_mask]

    # 4. Construct PyTorch Geometric Data Object
    graph_data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
        sens=torch.tensor(sens_attr, dtype=torch.float)
    )
    
    # 5. Save output file
    torch.save(graph_data, output_path)
    
    # Cleanup temporary files
    for f in [csv_path, rel_path]:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists("./temp_pokec"):
        os.rmdir("./temp_pokec")
    
    print(f"File Path: '{output_path}'")
    print(f"Nodes: {graph_data.num_nodes}")
    print(f"Edges: {graph_data.num_edges}")
    print(f"Features per node: {graph_data.num_features}")
    print(f"Sensitive Attribute (Žilina region): {int(graph_data.sens.sum().item())} positive instances")

if __name__ == "__main__":
    download_and_build_pokec_z(output_path="pokec_z_graph.pt", max_nodes=2000, sample_subgraph=True)