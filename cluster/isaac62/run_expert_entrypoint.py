"""Run the real evaluator's public main with SDK-version-aware test teardown.

Isaac 5.1 does not expose close(exit_code=...). A durable result written before
SDK shutdown lets the supervising probe distinguish errors from a zero SDK exit.
This does not patch the simulator or change evaluator/policy behavior.
"""
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import runpy
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrypoint', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = args.entrypoint.resolve(strict=True)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *arguments]
    result = dict(started=time.time(), completed=False, interpreter=sys.executable,
                  versions={name: importlib.metadata.version(name) for name in ('torch', 'isaacsim')})
    namespace = None
    try:
        namespace = runpy.run_path(str(script), run_name='__migration_entrypoint__')
        namespace['main']()
        result['completed'] = True
        module_name = 'isaaclab_tasks.manager_based.interactive_motion_tracking.g1.motion_tracking_pick_env'
        module = sys.modules.get(module_name)
        result['task_module'] = getattr(module, '__file__', None)
    except BaseException:
        result['error'] = traceback.format_exc()
        traceback.print_exc()
        raise
    finally:
        result['finished'] = time.time()
        args.receipt.write_text(json.dumps(result, indent=2)+'\n')
        if namespace and 'simulation_app' in namespace:
            app = namespace['simulation_app']
            kwargs = dict(wait_for_replicator=False)
            if 'exit_code' in inspect.signature(app.close).parameters:
                kwargs['exit_code'] = 0 if result['completed'] else 1
            app.close(**kwargs)


if __name__ == '__main__':
    main()
