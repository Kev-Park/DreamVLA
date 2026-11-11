"""GR00T inference across the simulator/model environment boundary.

The transport follows GR00T's endpoint/data MessagePack+NumPy convention. The
DreamVLA endpoints additionally round-trip RNG state and optional backbone
features so moving inference does not discard collection diagnostics or seeding.
No simulator or GR00T model package is imported by this module.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import random
import uuid

import msgpack_numpy as mnp
import numpy as np
import torch
import zmq

PROTOCOL_VERSION = 1


def _writable_arrays(value):
    # MessagePack's byte-backed arrays are read-only. The native policy receives
    # writable observation arrays and returns writable action arrays.
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _writable_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_writable_arrays(item) for item in value]
    return value


def capture_rng(device):
    device = torch.device(device)
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state().numpy().copy(),
            'torch_cuda': (torch.cuda.get_rng_state(device).cpu().numpy().copy()
                           if device.type == 'cuda' else None)}


def restore_rng(state, device):
    device = torch.device(device)
    if (state['torch_cuda'] is not None) != (device.type == 'cuda'):
        raise ValueError('Client and server must both use CUDA or both use CPU')
    # Validate Torch states before modifying the generators used for inference.
    cpu = np.asarray(state['torch_cpu'])
    if cpu.dtype != np.uint8 or cpu.ndim != 1:
        raise ValueError('Invalid CPU RNG state')
    cuda = None if device.type != 'cuda' else np.asarray(state['torch_cuda'])
    if cuda is not None and (cuda.dtype != np.uint8 or cuda.ndim != 1):
        raise ValueError('Invalid CUDA RNG state')
    py = state['python']
    random.setstate((py[0], tuple(py[1]), py[2]))
    np.random.set_state(tuple(state['numpy']))
    torch.set_rng_state(torch.from_numpy(cpu.copy()))
    if cuda is not None:
        torch.cuda.set_rng_state(torch.from_numpy(cuda.copy()), device)


def checkpoint_identity(path):
    path = Path(path).expanduser().resolve(strict=True)
    config = path/'config.json'
    return {'path': str(path), 'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest()}


class PolicyService:
    """One loaded policy, synchronous requests, and explicit client authorization."""

    def __init__(self, policy, checkpoint, device, embodiment_tag):
        self.policy = policy
        self.device = str(device)
        self.token = uuid.uuid4().hex
        self.identity = checkpoint_identity(checkpoint)
        self.session_id = uuid.uuid4().hex
        self.embodiment_tag = embodiment_tag
        self.running = True
        self._feature = None
        self._feature_hook = None

    def metadata(self):
        return {'protocol': PROTOCOL_VERSION, 'session_id': self.session_id,
                'checkpoint': self.identity, 'embodiment_tag': self.embodiment_tag,
                'pid': os.getpid(), 'device': self.device, 'torch': torch.__version__,
                'numpy': np.__version__,
                'has_backbone': getattr(getattr(self.policy, 'model', None), 'backbone', None) is not None}

    def _enable_features(self):
        if self._feature_hook is not None:
            return
        backbone = getattr(getattr(self.policy, 'model', None), 'backbone', None)
        if backbone is None:
            raise RuntimeError('VLA policy exposes no .model.backbone -- cannot capture features')

        def grab(_module, _inputs, output):
            value = output
            if not torch.is_tensor(value):
                value = (value.get('backbone_features') if hasattr(value, 'get')
                         else getattr(value, 'backbone_features', None))
            if torch.is_tensor(value):
                self._feature = value.detach().float().mean(dim=1)[0].cpu().numpy().astype(np.float16)

        self._feature_hook = backbone.register_forward_hook(grab)

    def dispatch(self, request):
        if not hmac.compare_digest(str(request.get('api_token', '')), self.token):
            raise PermissionError('Unauthorized inference-service request')
        if request.get('session_id') != self.session_id:
            raise ValueError('Inference-service session mismatch')
        endpoint, data = request['endpoint'], request.get('data', {})
        if endpoint == 'metadata':
            return self.metadata()
        if endpoint == 'shutdown':
            self.running = False
            return {'stopped': True}
        if endpoint not in ('get_action', 'reset'):
            raise ValueError('Unknown inference-service endpoint: '+str(endpoint))
        restore_rng(data['rng_state'], self.device)
        if endpoint == 'reset':
            result = self.policy.reset(data.get('options'))
            return {'result': result, 'rng_state': capture_rng(self.device)}
        self._feature = None
        if data.get('capture_features', False):
            self._enable_features()
        elif self._feature_hook is not None:
            self._feature_hook.remove()
            self._feature_hook = None
        action, info = self.policy.get_action(data['observation'], data.get('options'))
        return {'action': action, 'info': info, 'rng_state': capture_rng(self.device),
                'backbone_feature': self._feature}

    def serve(self, ready, endpoint='tcp://127.0.0.1:*'):
        if not endpoint.startswith('tcp://127.0.0.1:'):
            raise ValueError('Inference service must bind to the loopback interface')
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(endpoint)
            descriptor = {**self.metadata(), 'endpoint': socket.getsockopt_string(zmq.LAST_ENDPOINT),
                          'api_token': self.token}
            ready(descriptor)
            while self.running:
                request_id = None
                try:
                    request = _writable_arrays(mnp.unpackb(socket.recv(), raw=False))
                    request_id = request.get('request_id')
                    response = {'request_id': request_id, 'result': self.dispatch(request)}
                    message = mnp.packb(response)
                except Exception as exc:
                    logging.getLogger(__name__).exception('GR00T inference-service request failed')
                    message = mnp.packb({'request_id': request_id,
                                         'error': type(exc).__name__+': '+str(exc)})
                socket.send(message)
        finally:
            if self._feature_hook is not None:
                self._feature_hook.remove()
                self._feature_hook = None
            socket.close()
            context.term()


class PolicyServiceClient:
    """Preserve the get_action contract without importing GR00T into Isaac 6.2."""

    def __init__(self, descriptor, checkpoint, device, embodiment_tag='unitree_g1_sonic', timeout_ms=120000):
        if isinstance(descriptor, (str, Path)):
            descriptor = json.loads(Path(descriptor).read_text())
        self.descriptor = dict(descriptor)
        if descriptor['protocol'] != PROTOCOL_VERSION:
            raise ValueError('Unsupported inference-service protocol')
        if descriptor['checkpoint'] != checkpoint_identity(checkpoint):
            raise ValueError('Inference service checkpoint differs from requested checkpoint')
        if descriptor['embodiment_tag'] != embodiment_tag:
            raise ValueError('Inference service embodiment differs from requested embodiment')
        if not descriptor['endpoint'].startswith('tcp://127.0.0.1:'):
            raise ValueError('Use a local inference service on the same host')
        if torch.device(device).type != torch.device(descriptor['device']).type:
            raise ValueError('Client and server must both use CUDA or both use CPU')
        self.device = str(device)
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = None
        self._request_id = 0
        self._capture_features = False
        self.last_backbone_feature = None
        self._connect()
        try:
            metadata = self._call('metadata')
            for key in ('protocol', 'session_id', 'checkpoint', 'embodiment_tag'):
                if metadata[key] != descriptor[key]:
                    raise ValueError('Live inference-service identity differs from descriptor: '+key)
        except BaseException:
            self.close()
            raise

    def _connect(self):
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(self.descriptor['endpoint'])

    def _call(self, endpoint, data=None):
        self._request_id += 1
        request = {'endpoint': endpoint, 'data': data or {}, 'request_id': self._request_id,
                   'api_token': self.descriptor['api_token'], 'session_id': self.descriptor['session_id']}
        try:
            self.socket.send(mnp.packb(request))
            response = _writable_arrays(mnp.unpackb(self.socket.recv(), raw=False))
        except zmq.Again as exc:
            self._connect()
            raise TimeoutError('Inference service timed out; request was not automatically retried') from exc
        if response.get('request_id') != self._request_id:
            raise RuntimeError('Inference-service response does not match request')
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response['result']

    def enable_backbone_features(self):
        if not self.descriptor.get('has_backbone'):
            raise RuntimeError('Inference policy has no backbone feature interface')
        self._capture_features = True

    def get_action(self, observation, options=None):
        result = self._call('get_action', {'observation': observation, 'options': options,
                                         'capture_features': self._capture_features,
                                         'rng_state': capture_rng(self.device)})
        restore_rng(result['rng_state'], self.device)
        self.last_backbone_feature = result['backbone_feature']
        return result['action'], result['info']

    def reset(self, options=None):
        result = self._call('reset', {'options': options, 'rng_state': capture_rng(self.device)})
        restore_rng(result['rng_state'], self.device)
        self.last_backbone_feature = None
        return result['result']

    def shutdown_server(self):
        return self._call('shutdown')

    def close(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        if not self.context.closed:
            self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
