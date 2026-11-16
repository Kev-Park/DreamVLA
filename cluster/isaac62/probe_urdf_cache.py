"""Fault-inject a modern converter failure and verify real cache retry behavior."""
import argparse
import json
from pathlib import Path
import traceback


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, scope=__doc__, test_fixture_only=True)
    app = None
    try:
        app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app
        from unittest.mock import patch
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
        from isaaclab.sim.converters import urdf_converter_62
        source = args.output/'cache_fixture.urdf'
        source.write_text('<robot name="cache_fixture"><link name="base"/></robot>')
        cfg = UrdfConverterCfg(asset_path=str(source), usd_dir=str(args.output/'cache'),
            fix_base=True, force_usd_conversion=False)
        calls = []

        def injected(converter, config):
            calls.append(str(converter.usd_path))
            Path(converter.usd_path).write_text('#usda 1.0\ndef Xform "test_fixture" {}\n')
            if len(calls) == 1:
                raise RuntimeError('Deliberate converter post-process failure')

        with patch.object(urdf_converter_62, 'convert_asset', injected):
            try:
                UrdfConverter(cfg)
            except RuntimeError as error:
                assert str(error) == 'Deliberate converter post-process failure'
            else:
                raise AssertionError('Fault injection did not execute')
            assert (args.output/'cache/cache_fixture.usd').is_file()
            assert not (args.output/'cache/.asset_hash').exists(), 'Failure left a reusable cache marker'
            repaired = UrdfConverter(cfg)
            assert len(calls) == 2, 'Partial USD incorrectly reused after failure'
            UrdfConverter(cfg)
            assert len(calls) == 2, 'Successful cache was not reused'
        original_hash = AssetConverterBase._config_to_hash(cfg)
        assert repaired._config_to_hash(cfg) != original_hash, 'Old modern cache was not invalidated'
        legacy = object.__new__(UrdfConverter)
        legacy._modern_urdf_importer = False
        assert legacy._config_to_hash(cfg) == original_hash, 'Legacy cache behavior changed'
        report.update(completed=True, conversion_calls=len(calls), partial_output_retried=True,
            successful_output_reused=True, old_modern_hash_invalidated=True, legacy_hash_unchanged=True)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        (args.output/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True,
                exit_code=0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
