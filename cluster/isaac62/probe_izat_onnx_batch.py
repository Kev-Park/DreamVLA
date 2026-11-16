"""Compare batched release graphs with independent original ORT row calls."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import onnx
    import onnxruntime as ort
    from dreamvla_drawer_button.isaac.sonic_onnx import session_source

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    rows = []
    rng = np.random.default_rng(619)
    for filename, width in [('model_encoder.onnx', 1762), ('model_decoder.onnx', 994)]:
        path = args.models / filename
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        assert session_source(path, 1) == str(path)
        source = session_source(path, 2)
        assert isinstance(source, bytes)
        original_model = onnx.load(str(path))
        batch_model = onnx.load_model_from_string(source)
        initializers = {x.name: x.SerializeToString() for x in batch_model.graph.initializer}
        for value in original_model.graph.initializer:
            assert initializers[value.name] == value.SerializeToString(), value.name
        constants = {x.name: x.SerializeToString() for x in batch_model.graph.node if x.op_type == 'Constant'}
        for value in original_model.graph.node:
            if value.op_type == 'Constant':
                assert constants[value.name] == value.SerializeToString(), value.name
        original = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        batched = ort.InferenceSession(source, sess_options=options, providers=['CPUExecutionProvider'])
        name = original.get_inputs()[0].name
        checks = []
        for count in (1, 2, 7):
            data = rng.normal(0, 0.05, (count, width)).astype(np.float32)
            if width == 1762:
                data[:, :4] = 0
                # Include G1 mode (four zero flags) and independent other modes.
                for row in range(1, count):
                    data[row, (row - 1) % 4] = 1
            expected = np.concatenate([original.run(None, {name: row[None]})[0] for row in data])
            got = batched.run(None, {name: data})[0]
            assert np.isfinite(got).all()
            tolerance = 1e-6 if width == 1762 else 1e-5
            np.testing.assert_allclose(got, expected, rtol=tolerance, atol=tolerance)
            reverse = batched.run(None, {name: data[::-1].copy()})[0][::-1]
            np.testing.assert_allclose(reverse, got, rtol=tolerance, atol=tolerance)
            checks.append(dict(batch=count, maximum_error=float(np.max(np.abs(got-expected))),
                               exact=bool(np.array_equal(got, expected)), permutation_checked=True))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        rows.append(dict(model=filename, sha256=before, original_initializers_unchanged=True,
                         original_constant_nodes_unchanged=True, batches=checks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(completed=True, models=rows), indent=2)+'\n')
    print(args.output.read_text())


if __name__ == '__main__':
    main()
