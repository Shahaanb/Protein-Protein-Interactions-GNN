import torch
from model import ProteomeXGAT

def export_model(onnx_path="proteome_x_gat.onnx"):
    model = ProteomeXGAT(in_channels=64, hidden_channels=32, out_channels=1, heads=4)
    model.eval()

    # Create dummy data for tracing
    num_nodes = 50
    num_edges = 200
    x = torch.randn((num_nodes, 64))
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Export configuration
    torch.onnx.export(
        model,
        (x, edge_index, batch),
        onnx_path,
        dynamo=False,
        export_params=True,
        opset_version=18, # Required for scatter_reduce used in recent PyG/Torch ops
        do_constant_folding=True,
        input_names=['x', 'edge_index', 'batch'],
        output_names=['prob', 'node_attention_scores'],
        dynamic_axes={
            'x': {0: 'num_nodes'},
            'edge_index': {1: 'num_edges'},
            'batch': {0: 'num_nodes'},
            'node_attention_scores': {0: 'num_nodes'}
        }
    )
    print(f"Model effectively exported to {onnx_path}")

if __name__ == "__main__":
    export_model()
