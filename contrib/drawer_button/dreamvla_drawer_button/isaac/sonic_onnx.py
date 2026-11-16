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
    if shapes[0][1].dim_value == 1762:
        # The export also constructs a one-row mode mask with ScatterND. Merely
        # unfreezing reshapes makes different environments overwrite that same
        # row, selecting one environment's mode for the whole batch. Give each
        # environment its own zero-mask row and its own scatter row index.
        scatter = [node for node in graph.node if node.op_type == "ScatterND"]
        if len(scatter) != 1:
            raise ValueError("Expected the released SONIC mode-mask scatter")
        scatter = scatter[0]
        mask = constants.get(scatter.input[0])
        producers = {output: node for node in graph.node for output in node.output}
        indices = producers[scatter.input[1]]
        if (mask is None or mask.shape != (1, 3) or np.any(mask)
                or indices.op_type != "Concat" or len(indices.input) != 2):
            raise ValueError("Unexpected SONIC mode-mask construction")
        prefix = "dreamvla_mode_batch_"
        if any(name.startswith(prefix) for name in names):
            raise ValueError("Batch mode graph name collision")
        def tensor(name, data):
            result = prefix+name
            graph.initializer.append(numpy_helper.from_array(np.asarray(data, dtype=np.int64), result))
            return result
        zero = tensor("zero", 0)
        one = tensor("one", 1)
        axes0 = tensor("axes0", [0])
        axes1 = tensor("axes1", [1])
        three = tensor("width", [3])
        def node(op, inputs, output, **attributes):
            return onnx.helper.make_node(op, inputs, [prefix+output], name=prefix+output, **attributes)
        extra = [
            node("Shape", [graph.input[0].name], "input_shape"),
            node("Gather", [prefix+"input_shape", zero], "count", axis=0),
            node("Unsqueeze", [prefix+"count", axes0], "count_vector"),
            node("Concat", [prefix+"count_vector", three], "mask_shape", axis=0),
            node("ConstantOfShape", [prefix+"mask_shape"], "mask",
                 value=numpy_helper.from_array(np.zeros(1, dtype=mask.dtype))),
            node("Range", [zero, prefix+"count", one], "row_indices"),
            node("Unsqueeze", [prefix+"row_indices", axes1], "row_column"),
        ]
        indices.input[0] = prefix+"row_column"
        scatter.input[0] = prefix+"mask"
        # All inserted inputs originate from graph inputs/initializers, so they
        # can precede the unchanged original computation in topological order.
        original_nodes = list(graph.node)
        del graph.node[:]
        graph.node.extend(extra+original_nodes)
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
