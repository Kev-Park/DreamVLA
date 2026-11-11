"""Audit the merged project surface against both recorded branch parents.

This checks files, declared symbols/class members and CLI options, not simulator behavior.
Runtime/physics/rendering gates must be recorded separately.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess

PARENTS = {
    'g1': '0505979bf6058b772edb1e7436bb757461aba88f',
    'locomotion': 'bb9d874aa8e528bc3f5d53c0421fca4a2d4ff9b1',
}
PREFIXES = (
    'cluster/', 'TrajGen/holosoma_adapters/',
    'Training/scripts/',
    'Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/',
    'Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/motion_tracking/',
    'Training/source/isaaclab_tasks/isaaclab_tasks/utils/motion_lib/',
)


def surface(text):
    tree = ast.parse(text)
    symbols = set()

    def declared(body, prefix=''):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix+node.name
                symbols.add(name)
                if isinstance(node, ast.ClassDef):
                    declared(node.body, name+'.')
            elif prefix and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                symbols.update(prefix+target.id for target in targets if isinstance(target, ast.Name))
            elif prefix and isinstance(node, (ast.If, ast.Try)):
                declared(node.body, prefix)
                declared(node.orelse, prefix)
                if isinstance(node, ast.Try):
                    declared(node.finalbody, prefix)
                    for handler in node.handlers:
                        declared(handler.body, prefix)

    declared(tree.body)
    options = {a.value for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == 'add_argument' for a in node.args
               if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith('--')}
    return symbols, options


def audit(repo):
    findings = []; counts = {}; baseline_errors = []
    for label, commit in PARENTS.items():
        paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=repo, text=True).splitlines()
        paths = [p for p in paths if p.startswith(PREFIXES)]
        counts[label] = len(paths)
        for name in paths:
            target = repo / name
            if not target.is_file():
                findings.append(dict(parent=label, file=name, kind='missing_file')); continue
            if not name.endswith('.py'):
                continue
            old = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=repo).decode('utf8')
            try:
                old_symbols, old_options = surface(old)
            except SyntaxError as error:
                baseline_errors.append(dict(parent=label, file=name, error=str(error))); continue
            try:
                symbols, options = surface(target.read_text(encoding='utf8'))
            except SyntaxError as error:
                findings.append(dict(parent=label, file=name, kind='syntax_error', error=str(error))); continue
            for kind, missing in [('missing_symbol', old_symbols-symbols), ('missing_cli_option', old_options-options)]:
                if missing:
                    findings.append(dict(parent=label, file=name, kind=kind, missing=sorted(missing)))
    return dict(parents=PARENTS, project_file_counts=counts, findings=findings,
                baseline_syntax_errors=baseline_errors,
                scope='Project file/declared symbol/class member/CLI union only; this does not establish semantic or runtime equivalence')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = audit(Path(__file__).resolve().parents[1])
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, indent=2))
    if result['findings']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
