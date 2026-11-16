"""Batch the released SONIC graphs without changing weights or per-row operations."""

from pathlib import Path


def session_source(path: Path, num_envs: int):
    """Return the original file, or an in-memory batch-polymorphic release graph.

    Both release graphs advertise batch one. The encoder also bakes that one
    into Reshape constants (the existing walking Torch loader handles the same
    export restriction). Keep every original constant intact and give each
    affected Reshape its own shape initializer with an inferred leading size.
    Nothing is written over the supplied models.
    """
    if num_envs == 1:
        return str(path)
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("Expected a single-input, single-output SONIC release graph")
    shapes = [x.type.tensor_type.shape.dim for x in (graph.input[0], graph.output[0])]
    if any(len(shape) != 2 for shape in shapes):
        raise ValueError("Expected rank-two SONIC input and output")
    if (shapes[0][1].dim_value, shapes[1][1].dim_value) not in ((1762, 64), (994, 29)):
        raise ValueError("Unexpected SONIC release input/output widths")
    if all(shape[0].dim_param for shape in shapes):
        return str(path)
    if any(shape[0].dim_value != 1 for shape in shapes):
        raise ValueError("Only batch-one SONIC release exports require translation")

    constants = {x.name: numpy_helper.to_array(x) for x in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attribute.t)
    names = {x for node in graph.node for x in (*node.input, *node.output)}
    names.update(x.name for x in graph.initializer)
    for index, node in enumerate(graph.node):
        if node.op_type != "Reshape":
            continue
        shape = constants.get(node.input[1])
        if shape is None or shape.ndim != 1 or not len(shape) or int(shape[0]) != 1:
            continue
        if shape.dtype.kind not in "iu" or np.any(shape[1:] <= 0):
            raise ValueError("Unexpected SONIC reshape; cannot infer its batch safely")
        name = f"dreamvla_batch_shape_{index}"
        if name in names:
            raise ValueError("Batch shape initializer name collision")
        replacement = shape.copy()
        replacement[0] = -1
        graph.initializer.append(numpy_helper.from_array(replacement, name))
        node.input[1] = name
        names.add(name)
    for shape in shapes:
        shape[0].dim_param = "batch"
    # Intermediate shape annotations from the fixed-size export are metadata,
    # not operations. Let ORT infer them from the newly dynamic graph.
    del graph.value_info[:]
    onnx.checker.check_model(model)
    return model.SerializeToString()
