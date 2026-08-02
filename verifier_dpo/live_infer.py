"""
live, single-turn scoring adapter for verifier_dpo, matching verifier.infer.Verifier's
.score(task_goal, plan, obs_before, action) -> (q_value, advantage) interface EXACTLY, so
rollout/runner.py's run_learned_coordinated_episode() and coordinator/train_ppo.py work UNCHANGED
when this is passed in place of a verifier.infer.Verifier instance -- no coordinator/runner code
needed to be touched to try verifier_dpo as the coordinator's reward source.

verifier_dpo was never trained to output a (q_value, advantage) PAIR -- it outputs one scalar
episode-preference score. this adapter defines a reasonable mapping onto the interface the
coordinator's existing counterfactual-reward mechanism already expects, rather than changing that
mechanism:
  - q_value: the growing-prefix episode_score() (mean per-token score of the trajectory-so-far
    WITH this turn's action appended) -- same growing-prefix scoring pattern already used and
    validated in scripts/best_of_n_eval.py's _score_candidates_dpo(), which is what produced the
    coin win-rate lift (build 14). scores stay in-distribution because they use the SAME whole-
    episode text format (verifier_dpo/dataset.py's build_episode_text) the model trained on.
  - advantage: q_value - (this same object's PREVIOUS call's q_value) -- a one-step delta, the
    same definition verifier_v5's advantage used BEFORE the counterfactual-reward fix (see
    rollout/runner.py's run_learned_coordinated_episode docstring). runner.py's own counterfactual
    mechanism (advantage(chosen) - advantage(continue_counterfactual)) is what actually isolates
    the coordinator's leverage; this adapter only needs to supply a directionally sensible
    advantage, not solve same-state resolution itself.

IMPORTANT, disclose honestly rather than assume away: build 14's mapreader trace showed
verifier_dpo giving two OPPOSITE actions at the SAME state nearly identical scores (7.407 vs
7.392) -- the same same-state indifference that made verifier_v5's counterfactual reward flat
(nonzero_frac 0.11-0.14) across 5 PPO runs. Using verifier_dpo here is an honest TEST of whether
that limitation is verifier-specific or a property of this environment/reward-construction
approach more broadly, not an assumption that a "better" verifier (by Best-of-N or episode-AUC
standards) resolves it. Track nonzero_frac from the first iteration, same as coordinator_v5, and
be prepared for the same flat result -- that would still be a real, informative finding.

usage (drop-in for verifier.infer.Verifier):
    from verifier_dpo.live_infer import LiveVerifierDPO
    verifier = LiveVerifierDPO("verifier_dpo/checkpoints/finetune_v2/model.pt")
    q_value, advantage = verifier.score(task_goal, plan, obs_before, action)
"""

import torch
from transformers import AutoTokenizer

from verifier_dpo.dataset import build_episode_text
from verifier_dpo.model import BASE_MODEL, PreferenceScorer


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LiveVerifierDPO:
    """loads a trained verifier_dpo checkpoint once, then scores turns cheaply. NOT
    trajectory-agnostic like verifier.infer.Verifier -- because verifier_dpo's score is a
    property of the WHOLE episode-so-far (not a single isolated turn, see module docstring),
    callers MUST call reset_episode() at the start of each new episode and score() turns of that
    SAME episode in order, or the growing-prefix text will be wrong. rollout/runner.py's episode
    loop naturally does this (one Verifier/LiveVerifierDPO instance is typically reused across
    many episodes sequentially, calling .score() once per turn in order within each episode) --
    this constraint just needs an explicit reset between episodes, which callers using this
    outside runner.py must add."""

    def __init__(self, checkpoint_path: str, max_length: int = 2048, device: torch.device | None = None):
        self.device = device or _get_device()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = PreferenceScorer(freeze_backbone=True)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._traj: dict | None = None
        self._last_q_value = 0.0

    def reset_episode(self, task_goal: str, plan: str) -> None:
        """MUST be called once at the start of each new episode, before any score() calls for
        it -- clears the growing-prefix state so this episode's turns don't get appended after a
        previous episode's, and resets the one-step advantage baseline to 0 (matching
        run_learned_coordinated_episode's last_advantage=0.0 before turn 1)."""
        self._traj = {"task_goal": task_goal, "plan": plan, "turns": []}
        self._last_q_value = 0.0

    @torch.no_grad()
    def score(
        self, task_goal: str, plan: str, obs_before: str, action: str, commit: bool = True,
    ) -> tuple[float, float]:
        """returns (q_value, advantage) for a turn appended to the currently-tracked episode (see
        reset_episode()). task_goal/plan are accepted to match verifier.infer.Verifier's exact
        signature but are NOT re-read from these arguments after reset_episode() -- the tracked
        episode's own task_goal/plan are used, so this call must be scoring a turn of the SAME
        episode reset_episode() was last called for.

        commit=True (default): the scored (obs_before, action) is a REAL turn that was actually
        taken in the environment -- it's added to the tracked episode so the NEXT score() call's
        growing-prefix includes it, and _last_q_value updates for the next advantage delta.

        commit=False: the scored action is HYPOTHETICAL and was never stepped into the
        environment (e.g. rollout/runner.py's counterfactual-reward side-channel call, which asks
        "what would continue have done on this exact turn" purely to compute a baseline to
        subtract, see run_learned_coordinated_episode's docstring) -- the growing-prefix state and
        _last_q_value are left untouched, so a hypothetical never-taken action doesn't corrupt
        every subsequent real turn's score. added after catching this exact bug in review before
        any Lightning AI run: runner.py calls .score() with an identical signature for both the
        real chosen action and the counterfactual probe, so without this flag there would be no
        way to tell them apart and the counterfactual call would silently pollute the tracked
        episode text with a phantom turn."""
        if self._traj is None:
            raise RuntimeError(
                "LiveVerifierDPO.score() called before reset_episode() -- verifier_dpo's score "
                "depends on the whole episode-so-far, not just this one turn, so it needs to know "
                "when a new episode starts. call reset_episode(task_goal, plan) first."
            )

        next_step = len(self._traj["turns"]) + 1
        candidate_traj = dict(self._traj)
        candidate_traj["turns"] = self._traj["turns"] + [{
            "role": "worker", "step": next_step, "obs_before": obs_before, "action": action,
        }]
        text, _ = build_episode_text(candidate_traj)
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt",
        ).to(self.device)
        q_value = self.model.episode_score(enc["input_ids"], enc["attention_mask"]).item()
        advantage = q_value - self._last_q_value

        if commit:
            self._last_q_value = q_value
            self._traj["turns"].append({
                "role": "worker", "step": next_step, "obs_before": obs_before, "action": action,
            })
        return q_value, advantage
