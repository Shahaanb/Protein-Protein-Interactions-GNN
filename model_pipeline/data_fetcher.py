import tempfile
import requests
import torch
from torch_geometric.data import Data
import biotite.structure.io.pdb as pdb
import biotite.structure as struc

def fetch_pdb_and_build_graph(pdb_id: str, threshold: float = 8.0) -> Data:
    """Fetches a PDB file and converts it into a distance-based PyTorch Geometric graph."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Could not fetch PDB {pdb_id}")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb') as f:
        f.write(response.text)
        f.flush()
        pdb_file = pdb.PDBFile.read(f.name)
        structure = pdb.get_structure(pdb_file, model=1)
    
    # Filter C-alpha atoms
    ca_atoms = structure[structure.atom_name == "CA"]
    coords = torch.tensor(ca_atoms.coord, dtype=torch.float32)
    n_nodes = len(coords)
    
    if n_nodes == 0:
        raise ValueError(f"No C-alpha atoms found for PDB {pdb_id}")

    # Node features: normally sequence embeddings (ProtBERT/ESM), we use random dummy features here
    x = torch.randn((n_nodes, 64)) 
    
    # Calculate pairwise distances and build edges
    distances = torch.cdist(coords, coords)
    
    # Create adjacency matrix masks (excluding self loops if you want, but GAT can handle them)
    adj_mask = distances < threshold
    
    # Get edge indices
    edge_index = adj_mask.nonzero(as_tuple=False).t().contiguous()
    
    data = Data(x=x, pos=coords, edge_index=edge_index)
    return data

if __name__ == "__main__":
    # Test script
    try:
        data = fetch_pdb_and_build_graph("1BRS")
        print(f"Graph for 1BRS built: {data.num_nodes} nodes, {data.num_edges} edges.")
    except Exception as e:
        print(f"Error: {e}")
