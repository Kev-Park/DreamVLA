# Limitations and experiment boundaries

## Release validation

The contribution is isolated under `contrib/drawer_button`, with unique task IDs,
so it does not overwrite Kevin's native task registrations. The extracted native
task snapshots still depend on DreamControl's common motion library, reset,
observation and reward functions. A change in those functions can change the
experiment even when this directory is unchanged. Pin revisions and run the GPU
smoke checklist in `setup.md` before merging or reporting new performance.

CPU tests verify interfaces and failure guards, not stable whole-body execution,
task success, camera asset fidelity or GPU stream behavior. No success-rate
claim accompanies the refactor. Historical model checkpoints, statistics,
recordings and evaluation outputs remain outside this code contribution.

## Fine-tuning is explicit, not retroactively identical

The native launcher uses NVIDIA's online dataset/trainer. Historical experiments
also used frozen visual-feature caches, sliced embodiment adapters, action-input
rescaling, warm starts and small-data refinement. Those are materially different
recipes. Their dated orchestration scripts are deliberately not presented as one
canonical pipeline here. The native launcher does not silently copy trained
weights, apply the x64 initialization, remove padded channels, or override the
standard loss/noise. An optional head checkpoint can be loaded explicitly for
inference with strict row/shape checks; that is not a fresh-pretrained training
run. `action_initialization.py` is an opt-in research utility, never auto-called.

Frame-epoch budgets, valid-window passes and optimizer steps are different
quantities; see `protocol.md`. Native sharded sampling has its own semantics.
The final effective batch can slightly exceed the nominal frame budget.

## DAgger teacher-history issue (not fixed by renaming a recipe)

The PPO teacher observes its previous **residual** action (66 dimensions). The
student predicts a full FSQ token plus 14 hand commands (78 dimensions). Their
history inputs are not interchangeable. The approximate residual reconstruction

```text
residual_proxy = (student_full_token / reference_token - 1) / alpha
```

cannot represent a nonzero student token where the reference is zero, and FSQ
rounding discards information about the original pre-quantized residual.
`actions.previous_action_proxy` returns the unrepresentable mask explicitly. It
is not an exact reconstruction, a validated correction, or equivalent to an
expert designed around decoded joint-command history.

`dagger.replay_plan` implements bounded attempts on failed student rollouts with
random takeover times. `dagger.aggregate` rejects duplicate demonstrations and
requires full-state expert labels for correction episodes. These bookkeeping
utilities do not resolve the history mismatch, reproduce simulator snapshots, or
make a teacher capable of recovering arbitrary off-distribution states.

There is deliberately **no automatic multi-round DAgger launcher** in this
candidate. Before adding one: define compatible teacher history, verify expert
action execution at takeover, label every student-visited training observation
(or use a mask-aware trainer), audit reset/replay equivalence, isolate final test
banks from correction selection, and compare to matched offline data/training
exposure. A success-only full-state replay recipe is a selective variant, not
unfiltered textbook DAgger. Fresh retraining and continuing prior weights must
also be reported as different optimization protocols.

## Evaluation limits

The drawer recorder's second track is currently seed variation, not additional
pose randomization. The standing-button second track includes small root
perturbations. Neither is a new out-of-distribution task or a real-robot test.
The 5 cm drawer acceptance criterion and 15 cm criterion are both retained and
must never be conflated. No MPJPE averaging convention is silently assumed.

The loopback bridge is for two trusted local processes, not an authenticated
public inference service. Keep it on `127.0.0.1`. Never load untrusted reference
pickles, PPO checkpoints or model files. No training/launch command grants
permission to monopolize shared GPUs; allocation remains an external concern.
