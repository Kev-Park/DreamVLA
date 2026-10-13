"""Walking GR00T chains: exact expert bases, three DAgger rounds, held-out evals.

Collection manifests fix the original reference IDs, independent of how many
passing episodes came from each reference. All fine-tunes independently start
from GR00T-N1.7-3B with batch 16 and ceil(5.7 * cumulative_frames / 16) updates.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import gr00t_budget_chain as base
import watch_t20max_v3 as common

base.REFS = "StitchMAX200v3"
base.EVAL_REFS = "StitchMAX100v3_eval"
ROOT, HERE = base.ROOT, base.HERE


class WalkChain(base.Chain):
    def __init__(self, budget):
        super().__init__(budget, tag=f"walk{budget}_v3_dagger57_20261013")
        self.expert = (ROOT/"eval_videos/T20_max_v3_env2048/early_stop/best_evaluated.pt").resolve(strict=True)
        assert self.expert.name == "model_4500.pt"
        self.rollout_steps = 1200
        self.training_cwd = HERE.parent/"Training"
        self.score_script = HERE/"walk_gr00t_score.py"
        self.env = common.environment(0)
        self.env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                        NUMEXPR_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
        self.env["PYTHONPATH"] = str(self.sim)

    def prepare(self):
        directory = ROOT/f"datasets/walk_v3_study/base{self.budget}"
        self.log(f"WAIT exact {self.budget}-episode expert base: {directory}")
        while not (directory/"ready").is_file():
            status = ROOT/"eval_videos/walk_v3_study/expert_collection/workflow.json"
            if status.exists() and base.read(status).get("status") == "failed":
                raise RuntimeError("Expert collection failed; inspect "+str(status))
            base.atomic(self.work/"waiting_for_base.json", {
                "time": time.time(), "pid": os.getpid(), "path": str(directory)})
            time.sleep(30)
        manifest = base.read(directory/"manifest.json")
        self.sources = [Path(r["path"]) for r in manifest["episodes"]]
        assert len(self.sources) == len(set(self.sources)) == self.budget
        assert all(p.is_file() and p.stat().st_size > 0 for p in self.sources)
        self.ids = manifest["selected_reference_ids"]
        assert len(self.ids) == len(set(self.ids)) == self.budget
        assert len(list((ROOT/"ref_motions"/base.REFS).glob("*.pkl"))) == 200
        assert len(list((ROOT/"ref_motions"/base.EVAL_REFS).glob("*.pkl"))) == 100
        base.atomic(self.work/"box.json", manifest["box"])
        demo = self.work/"demo_root"
        demo.mkdir(exist_ok=True)
        # Different seeded successes may have the same original motion ID.
        for i, source in enumerate(self.sources):
            link = demo/f"ep_{i:05d}__{source.name}"
            if not link.exists():
                link.symlink_to(source)
            assert link.resolve() == source.resolve()
        (self.work/"motion_ids.txt").write_text(" ".join(map(str, self.ids))+"\n")
        data = directory/"lerobot/ds"
        built = directory/"built.json"
        if not built.is_file():
            # Convert once into a single dataset; merge regenerates global stats.
            args = ["bash", HERE/"convert_sharded.sh", "--hdf5-root", directory/"hdf5",
                    "--recursive", "--output-path", directory/"lerobot", "--dataset-name", "ds",
                    "--task-prompt", "pick up the mustard bottle", "--fps", "50", "--shards", "4"]
            self.execute(args, "convert_base")
            info = base.read(data/"meta/info.json")
            assert info["total_episodes"] == self.budget and not info.get("discarded_episode_indices")
            base.atomic(built, {"episodes": self.budget, "frames": info["total_frames"]})
        info = base.read(data/"meta/info.json")
        assert info["total_episodes"] == self.budget
        ordered = sorted((directory/"hdf5").glob("*.hdf5"))
        assert [p.resolve() for p in ordered] == [p.resolve() for p in self.sources]
        base.atomic(self.work/"recipe.json", {
            "budget": self.budget, "base_dataset": str(data), "base_episodes": self.budget,
            "distinct_base_references": manifest["distinct_references"], "motion_ids": self.ids,
            "expert": str(self.expert), "rounds": 3, "student_attempts_per_round": self.budget,
            "batch_size": 16, "passes": 5.7, "steps_rule": "ceil(5.7*cumulative_frames/16)",
            "initialization": "independent nvidia/GR00T-N1.7-3B each round", "learning_rate": 0.0001,
            "reference_pool": base.REFS, "eval_pool": base.EVAL_REFS, "eval_episodes": 100,
            "rollout_cap": self.rollout_steps, "eval_no_ee_term": self.eval_no_ee_term,
            "collection": "one student attempt per selected reference; existing paired expert rescue; retain four-rule passes",
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()})
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, choices=[10, 50, 200], required=True)
    args = parser.parse_args()
    chain = WalkChain(args.budget)
    try:
        chain.run()
    except BaseException as exc:
        base.atomic(chain.work/"abort.json", {"time": time.time(), "error": repr(exc), "pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
