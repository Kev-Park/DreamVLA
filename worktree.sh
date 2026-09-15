#!/bin/bash
# worktree.sh -- one command per git worktree, nothing to remember afterwards.
#
#   ./worktree.sh new <branch> [holosoma-branch]   create ~/kevin/wt/<branch>/{DreamVLA,.venv[,holosoma]}
#   ./worktree.sh rm  <branch>                     remove it (worktrees + overlay venv)
#   ./worktree.sh ls                               list worktrees of this repo and of holosoma
#
# What a worktree gets:
#   DreamVLA/   git worktree of <branch> (created from the current HEAD if it does not exist yet)
#   .venv/      ~30 MB overlay on the dreamcontrol_51 conda env (--system-site-packages) with THIS
#               checkout's Training/source/* and Training/isaac_utils installed editable, so its
#               edits win over the main checkout's editable installs. Everything else (torch,
#               isaacsim, gear_sonic, gr00t, ...) is inherited from dreamcontrol_51 untouched.
#   holosoma/   optional paired worktree of the holosoma fork. Scripts resolve holosoma via
#               sibling(): a paired worktree beside DreamVLA wins, otherwise the shared
#               ~/kevin/holosoma is used. So only pass a holosoma branch when the work touches
#               the retargeter.
#
# Then:   source ~/kevin/wt/<branch>/.venv/bin/activate && cd ~/kevin/wt/<branch>/DreamVLA/Training
# and run scripts exactly as in the main checkout. Logs land under the worktree's Training/logs
# (they are CWD-relative); reference datasets are pooled in ~/kevin/ref_motions so every checkout sees them.
#
# Env: WT_ROOT (default ~/kevin/wt), DREAMCONTROL_PY (default: dreamcontrol_51's python).
set -euo pipefail
MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT_ROOT=${WT_ROOT:-~/kevin/wt}; WT_ROOT=$(eval echo "$WT_ROOT")
SHARED_ROOT=~/kevin
# same sibling() as TrajGen/holosoma_adapters/*.sh and isaaclab_tasks/utils/repo_paths.py
sibling() { local n=$1 ov; ov=$(eval echo "\$$(echo "$n" | tr 'a-z-' 'A-Z_')_DIR" 2>/dev/null || true)
  [ -n "$ov" ] && { eval echo "$ov"; return; }
  for r in "$(dirname "$MAIN")" "$SHARED_ROOT" ~; do [ -e "$r/$n" ] && { echo "$r/$n"; return; }; done
  echo "$(dirname "$MAIN")/$n"; }
EDITABLES=(Training/source/isaaclab Training/source/isaaclab_assets Training/source/isaaclab_mimic
           Training/source/isaaclab_rl Training/source/isaaclab_tasks Training/isaac_utils)

find_py() {
  [ -n "${DREAMCONTROL_PY:-}" ] && { echo "$DREAMCONTROL_PY"; return; }
  for p in ~/miniconda3/envs/dreamcontrol_51/bin/python ~/kevin/*/envs/dreamcontrol_51/bin/python; do
    [ -x "$p" ] && { echo "$p"; return; }
  done
  echo "worktree.sh: cannot find dreamcontrol_51 python (set DREAMCONTROL_PY)" >&2; exit 1
}

cmd_new() {
  local br=$1 hsbr=${2:-} wt
  wt="$WT_ROOT/$br"
  [ -e "$wt" ] && { echo "worktree.sh: $wt already exists (rm it first)" >&2; exit 1; }
  mkdir -p "$wt"
  echo "== DreamVLA worktree: $wt/DreamVLA  [$br]"
  if git -C "$MAIN" show-ref --verify -q "refs/heads/$br"; then
    git -C "$MAIN" worktree add "$wt/DreamVLA" "$br"
  elif git -C "$MAIN" show-ref --verify -q "refs/remotes/origin/$br"; then
    git -C "$MAIN" worktree add --track -b "$br" "$wt/DreamVLA" "origin/$br"
  else
    echo "   (branch $br does not exist; creating it from $(git -C "$MAIN" rev-parse --short HEAD))"
    git -C "$MAIN" worktree add -b "$br" "$wt/DreamVLA"
  fi
  if [ -n "$hsbr" ]; then
    local hs; hs=$(sibling holosoma)
    echo "== holosoma worktree: $wt/holosoma  [$hsbr]  (from $hs)"
    if git -C "$hs" show-ref --verify -q "refs/heads/$hsbr"; then
      git -C "$hs" worktree add "$wt/holosoma" "$hsbr"
    else
      git -C "$hs" worktree add -b "$hsbr" "$wt/holosoma"
    fi
  fi
  local py; py=$(find_py)
  echo "== overlay venv: $wt/.venv  (on $py)"
  "$py" -m venv --system-site-packages "$wt/.venv"
  for pkg in "${EDITABLES[@]}"; do
    "$wt/.venv/bin/pip" install -q -e "$wt/DreamVLA/$pkg" --no-deps --no-build-isolation
    echo "   editable: $pkg"
  done
  echo "== done. Use it with:"
  echo "   source $wt/.venv/bin/activate && cd $wt/DreamVLA/Training"
}

cmd_rm() {
  local br=$1 wt
  wt="$WT_ROOT/$br"
  [ -d "$wt/DreamVLA" ] && git -C "$MAIN" worktree remove --force "$wt/DreamVLA" && echo "removed $wt/DreamVLA"
  if [ -d "$wt/holosoma" ]; then
    git -C "$(sibling holosoma)" worktree remove --force "$wt/holosoma" && echo "removed $wt/holosoma"
  fi
  rm -rf "$wt/.venv"; rmdir "$wt" 2>/dev/null || true
  git -C "$MAIN" worktree prune
}

cmd_ls() {
  echo "== DreamVLA ($MAIN)"; git -C "$MAIN" worktree list | sed 's/^/   /'
  local hs; hs=$(sibling holosoma)
  [ -d "$hs/.git" ] && { echo "== holosoma ($hs)"; git -C "$hs" worktree list | sed 's/^/   /'; }
}

case "${1:-}" in
  new) [ -n "${2:-}" ] || { echo "usage: worktree.sh new <branch> [holosoma-branch]" >&2; exit 1; }; cmd_new "$2" "${3:-}" ;;
  rm)  [ -n "${2:-}" ] || { echo "usage: worktree.sh rm <branch>" >&2; exit 1; }; cmd_rm "$2" ;;
  ls)  cmd_ls ;;
  *)   sed -n '2,25p' "$0"; exit 1 ;;
esac
