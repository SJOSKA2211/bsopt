import os

import onnx
from onnx import TensorProto, helper


def create_minimal_onnx(output_path, input_dim=9):
    # Create an input (name: "input", type: float, shape: [batch_size, input_dim])
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch_size", input_dim])

    # Create an output (name: "output", type: float, shape: [batch_size, 1])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch_size", 1])

    # Weights constant [input_dim, 1]
    weights_data = [0.1] * input_dim
    weights_initializer = helper.make_tensor(
        name="weights", data_type=TensorProto.FLOAT, dims=[input_dim, 1], vals=weights_data
    )

    # Node: MatMul
    node_def = helper.make_node(
        "MatMul",
        ["input", "weights"],
        ["output"],
    )

    # Create the graph
    graph_def = helper.make_graph(
        [node_def],
        "test-model",
        [X],
        [Y],
        [weights_initializer],
    )

    # Create the model with explicit opset
    # We use opset 15 as it's very stable
    opset = helper.make_operatorsetid("", 15)

    model_def = helper.make_model(graph_def, producer_name="onnx-example", opset_imports=[opset])

    # Save the model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save(model_def, output_path)
    print(
        f"[+] Created minimal ONNX model (Opset 15) at {output_path} ({os.path.getsize(output_path)} bytes)"
    )


if __name__ == "__main__":
    create_minimal_onnx("models/latest_pricing.onnx", 9)
    create_minimal_onnx("models/latest_nn_pricing.onnx", 9)
    create_minimal_onnx("models/latest_xgb_pricing.onnx", 9)
