# Verifier v1

**Date:** 2026-07-13

**Overview:** First verifier checkpoint actually trained, downloaded, and confirmed working end to end. Qwen2.5-3B backbone plus a 2-output value head, trained on build_10's combined cross-environment data to predict Q-value and Advantage per worker turn. Trained on Lightning AI, full backbone unfrozen, on a 48GB GPU.

**Data:**
- data/labeled/build_10_combined (500 ALFWorld + 500 TextWorldExpress episodes, 27,459 labeled worker turns)
- Cross-environment on purpose: verifier learns from two different action-vocabulary profiles, not just ALFWorld

**Training setup:**
- Qwen2.5-3B-Instruct backbone, full fine-tune (not frozen)
- 8-bit AdamW (bitsandbytes) and gradient checkpointing, same memory-fitting recipe validated on the earlier failed training attempts
- Dual loss: L_Q + 0.5 * L_A
- Batch size 16 (fit comfortably on the 48GB GPU, versus batch size 4 needed on a 22GB card previously), learning rate 1e-5, 4 epochs
- Checkpoint downloaded from the Studio as a zip, extracted locally into checkpoints/verifier_v2 (naming note below)

**Validation (live inference, not just training loss):**
- Built rollout/infer.py's Verifier class, the first time this project has scored a turn with the verifier outside of training
- On a real won ALFWorld episode ("put some pencil on shelf"), Q-value climbed steadily: 0.169 to 0.946 at the final winning action
- On a real lost ALFWorld episode ("put a hot mug in cabinet"), Q-value stayed low throughout, including a run of examine coffeemachine 1 repeated 5 times where the score stayed pinned at 0.053 the entire time, and flatlined again during a 50-step-cap open/close cabinet loop
- Confirmed via a 60-episode sample of data/labeled/build_10_alfworld scored with the trained verifier: won-episode turns have p10=0.743, median=0.976; lost-episode turns have median=0.030, p75=0.056. Clean separation between the two distributions, no meaningful overlap at the percentile level.

**Notes:**
- This is the first verifier checkpoint this project has actually used for anything beyond a training-loss number. Everything downstream (the coordinator's live verifier-driven mode) depends on this checkpoint existing locally and scoring correctly, both of which are now confirmed.
- Naming inconsistency worth flagging: the checkpoint directory on disk is checkpoints/verifier_v2, even though this is the first verifier that was actually retrieved and used (an earlier "v1" training run's weights were never downloaded and are considered lost). This report calls the model itself v1 since it is the first real one, but the directory name does not match. Worth reconciling before a v3 exists, to avoid the same confusion compounding.
- Initial live-inference threshold calibration for the coordinator (see rollout/coordinator.py) was picked from a single hand-inspected trajectory and turned out to be wrong once tested against the real percentile data above. The 60-episode sample here is what caught and fixed that.
- Next: this checkpoint is what the coordinator's verifier-driven mode (continue/retry/replan/backtrack/escalate) is built against. Any real data collection using it should run on a GPU, not locally, since running the worker and the verifier as two separate ~3B models on Apple Silicon is far too slow for anything beyond a short smoke test.
