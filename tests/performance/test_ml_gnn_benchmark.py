import time

import torch

from src.ml.reinforcement_learning.gnn_policy import GNNFeatureExtractor


def test_gnn_feature_extractor_latency_benchmark():
    # Setup
    input_dim = 16
    output_dim = 64
    extractor = GNNFeatureExtractor(input_dim=input_dim, output_dim=output_dim)
    extractor.eval() # Inference mode
    
    # Create a reasonably sized graph (e.g. 50 options in a chain)
    num_nodes = 50
    x = torch.randn(num_nodes, input_dim)
    
    # Create fully connected edges for worst-case complexity
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            extractor(x, edge_index)
            
    # Benchmark
    num_runs = 100
    start_time = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            extractor(x, edge_index)
    duration = time.perf_counter() - start_time
    
    avg_latency_ms = (duration / num_runs) * 1000
    print(f"\nGNN Inference Latency (50 nodes, fully connected): {avg_latency_ms:.2f} ms")
    
    # We want sub-5ms for real-time arbitrage
    assert avg_latency_ms < 5.0

def test_gnn_throughput_massive_batch():
    input_dim = 16
    output_dim = 64
    extractor = GNNFeatureExtractor(input_dim=input_dim, output_dim=output_dim)
    
    # Simulate a massive batch of graphs
    num_graphs = 32
    num_nodes_per_graph = 20
    x = torch.randn(num_graphs * num_nodes_per_graph, input_dim)
    
    # Simple chain edges
    edges = []
    for g in range(num_graphs):
        offset = g * num_nodes_per_graph
        for i in range(num_nodes_per_graph - 1):
            edges.append([offset + i, offset + i + 1])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Batch vector
    batch = torch.tensor([i // num_nodes_per_graph for i in range(num_graphs * num_nodes_per_graph)])
    
    start_time = time.perf_counter()
    features = extractor(x, edge_index, batch=batch)
    duration = time.perf_counter() - start_time
    
    print(f"GNN Throughput ({num_graphs} graphs batch): {duration*1000:.2f} ms")
    assert features.shape == (num_graphs, output_dim)
