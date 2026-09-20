"""Parse eval_vla_sonic.py logs into the persistent results store.

    python summarize_vla_eval.py --tag dagger2_adroit --logs out/dagger2_adroit_eval.log \
        [--notes "20 eps, HS_EVAL_NO_EE_TERM=1, 4-way split"]

Writes two CSVs under ``results/vla_eval/``:
  <tag>_episodes.csv   one row per episode (motion, held flags, lift, topple, grasp timing)
  summary.csv          one row per run, rewritten in place when a tag is re-summarised

Why a store: eval numbers otherwise live only in cluster /tmp logs, which are deleted -- that is
how the provenance of an earlier "45% held" figure was lost. The headline column is
``held5_upright``: ``held`` alone counts episodes in which the bottle toppled but was still lifted,
which overstates success (see eval_vla_sonic.py's [PHYS] held vs held-upright).
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
STORE = REPO / "results" / "vla_eval"

EP_RE = re.compile(
    r"\[episode (?P<ep>\d+)\] motion_id=(?P<motion>\d+)(?P<body>.*?)"
    r"ended at step (?P<steps>\d+) \((?P<end>\w+)\)\s+"
    r"phys_held\(2cm/5cm\)=(?P<h2>True|False)/(?P<h5>True|False)\s+"
    r"max_lift=(?P<lift>[\d.]+)cm\s+touched=(?P<touched>True|False) toppled=(?P<toppled>True|False)",
    re.S,
)
GRASP_RE = re.compile(r"first close step: (\d+)\s+ref grab step: (\d+)")


def parse(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(paths):
        text = Path(p).read_text(errors="ignore")
        for m in EP_RE.finditer(text):
            tail = text[m.end(): m.end() + 800]
            g = GRASP_RE.search(tail)
            close, ref = (int(g.group(1)), int(g.group(2))) if g else (None, None)
            rows.append(
                dict(
                    motion=int(m.group("motion")),
                    steps=int(m.group("steps")),
                    end=m.group("end"),
                    held2=m.group("h2") == "True",
                    held5=m.group("h5") == "True",
                    max_lift_cm=float(m.group("lift")),
                    touched=m.group("touched") == "True",
                    toppled=m.group("toppled") == "True",
                    close_step=close,
                    ref_grab_step=ref,
                    close_lag=None if close is None else close - ref,
                )
            )
    rows.sort(key=lambda r: r["motion"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="run name, e.g. dagger2_adroit")
    ap.add_argument("--logs", nargs="+", required=True, help="eval log file(s) or globs")
    ap.add_argument("--notes", default="", help="free-text: settings that make the run comparable")
    args = ap.parse_args()

    paths = [p for g in args.logs for p in (glob.glob(g) or [g])]
    rows = parse(paths)
    if not rows:
        raise SystemExit(f"no episodes parsed from {paths}")

    STORE.mkdir(parents=True, exist_ok=True)
    ep_csv = STORE / f"{args.tag}_episodes.csv"
    with ep_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    lifts = sorted(r["max_lift_cm"] for r in rows)
    lags = [r["close_lag"] for r in rows if r["close_lag"] is not None]
    summary = dict(
        tag=args.tag,
        episodes=n,
        held2=sum(r["held2"] for r in rows),
        held5=sum(r["held5"] for r in rows),
        # the honest headline: lifted AND still upright
        held5_upright=sum(r["held5"] and not r["toppled"] for r in rows),
        toppled=sum(r["toppled"] for r in rows),
        touched=sum(r["touched"] for r in rows),
        lift_median_cm=round(st.median(lifts), 2),
        lift_max_cm=round(lifts[-1], 2),
        close_lag_median=int(st.median(lags)) if lags else "",
        notes=args.notes,
    )

    sum_csv = STORE / "summary.csv"
    existing = []
    if sum_csv.exists():
        with sum_csv.open(newline="") as fh:
            existing = [r for r in csv.DictReader(fh) if r["tag"] != args.tag]
    with sum_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary))
        w.writeheader()
        w.writerows(existing + [{k: str(v) for k, v in summary.items()}])

    print(f"[summarize] {ep_csv.relative_to(REPO)}  ({n} episodes)")
    print(f"[summarize] {sum_csv.relative_to(REPO)}")
    print("  " + "  ".join(f"{k}={v}" for k, v in summary.items() if k != "notes"))


if __name__ == "__main__":
    main()
