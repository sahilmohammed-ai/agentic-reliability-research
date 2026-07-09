# Verifier v1 Training

**Date:** 2026-07-08

**Overview:** First full fine-tune of the verifier. Qwen2.5-3B backbone plus a 2-output value head, trained on build_5's labeled data to predict Q-value and Advantage per worker turn. Trained on Lightning AI (single GPU, ~22GB VRAM), full backbone unfrozen, not just the head.

**Data:**
- data/labeled/build_5 (500 trajectories, 22,124 labeled worker turns)
- 19,912 train examples, 2,212 val examples

**Training setup:**
- Qwen2.5-3B-Instruct backbone, full fine-tune (not frozen)
- 8-bit AdamW (bitsandbytes) and gradient checkpointing, both needed to fit on a single GPU
- Dual loss: L_Q + 0.5 * L_A
- Batch size 4, learning rate 1e-5, 5 epochs

**Metrics (val loss by epoch):**
- Epoch 1: q_loss 0.0185, adv_loss 0.0022
- Epoch 2: q_loss 0.6483, adv_loss 0.0460 (spike, model recovered next epoch)
- Epoch 3: q_loss 0.0330, adv_loss 0.0004
- Epoch 4: q_loss 0.0326, adv_loss 0.0001 (best epoch)
- Epoch 5: q_loss 0.0687, adv_loss 0.0067 (worse than epoch 4)

**Notes:**
- Model converged by epoch 3-4. Epoch 5 was not an improvement, val loss roughly doubled versus epoch 4.
- Two loss spikes during training (epoch 2, and a smaller one late in epoch 5), both self-recovered by the next epoch. Cause not confirmed, possibly the learning rate being briefly too aggressive for a batch or sequence of batches.
- Training script only saved the last epoch's weights, not the best one, so the first full run's saved checkpoint was actually the worse epoch 5 weights. Fixed in the script afterward: it now tracks best validation loss across the run and saves that separately as verifier.pt, with the last epoch kept as verifier_last.pt for comparison.
- Getting a clean run took three attempts, each hitting a different memory issue: plain AdamW's optimizer state didn't fit (fixed with 8-bit AdamW), then activations during the forward pass didn't fit (fixed with gradient checkpointing), then validation OOM'd due to a bug where the model tracked gradients during eval too (fixed by checking train/eval mode correctly).
