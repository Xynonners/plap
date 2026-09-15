#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "arc-agi==0.9.9",
#   "arc-agi-3-benchmarking @ git+https://github.com/arcprize/arc-agi-3-benchmarking.git@86d72170ce3155551712a9fafd290bab471d6eee",
#   "openai==2.41.1",
#   "python-dotenv>=1.1,<2",
# ]
# ///
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from arc_agi import Arcade, OperationMode
from arc_agi.scorecard import EnvironmentScorecard
from arcengine import FrameData
from benchmarking.agent import BenchmarkingAgent
from benchmarking.base import ExitReason
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARC_DIR = REPO_ROOT / ".dev" / "arc"
DEFAULT_ARC_ROOT_URL = "https://three.arcprize.org"
MODEL_CONFIG_ID = "plap-local"

ModelConfig = tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, float],
]


class LocalBenchmarkingAgent(BenchmarkingAgent):
    """Official benchmarking agent with local model config and smoke stopping."""

    def _load_model_config(self) -> ModelConfig:
        return self._local_model_config

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if self._action_limit is not None and self.action_counter >= self._action_limit:
            self.exit_reason = ExitReason.ACTION_BUDGET
            return True
        return super().is_done(frames, latest_frame)

    def __init__(
        self,
        *args: Any,
        local_model_config: ModelConfig,
        action_limit: int | None,
        **kwargs: Any,
    ) -> None:
        self._local_model_config = local_model_config
        self._action_limit = action_limit
        super().__init__(*args, **kwargs)
        if action_limit is not None:
            self.MAX_ACTIONS = action_limit


def _load_environment() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(REPO_ROOT / ".dev" / ".env", override=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official ARC-AGI-3 benchmarking agent locally through PLAP.")
    parser.add_argument("--game", default="ls20", help="ARC-AGI-3 game ID (default: ls20).")
    parser.add_argument("--model", default=None, help="PLAP model ID (default: PLAP_DEV_MODEL or plap-ai/mote).")
    parser.add_argument("--base-url", default=None, help="PLAP OpenAI base URL (default: PLAP_DEV_BASE_URL).")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use downloaded environment files without ARC discovery or downloads.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=None,
        help="Stop after this many submitted actions; omit for official per-level budgets.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=128_000)
    parser.add_argument("--max-context-chars", type=int, default=175_000)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()
    if args.max_actions is not None and args.max_actions < 1:
        parser.error("--max-actions must be at least 1")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be at least 1")
    if args.max_context_chars < 1:
        parser.error("--max-context-chars must be at least 1")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    return args


def _model_config(args: argparse.Namespace, base_url: str, model: str) -> ModelConfig:
    return (
        {
            "MAX_ACTIONS_BASELINE_MULTIPLIER": 5.0,
            "MAX_CONTEXT_LENGTH": args.max_context_chars,
            "MAX_RETRIES": args.max_retries,
            "analysis_mode": False,
        },
        {
            "sdk": "openai-python",
            "api": "responses",
            "state": "manual_rolling",
        },
        {
            "base_url": base_url,
            "api_key_env": "PLAP_DEV_API_KEY",
        },
        {
            "model": model,
            "max_output_tokens": args.max_output_tokens,
        },
        {},
    )


def _close_scorecard(
    arcade: Arcade,
    card_id: str,
    agent: LocalBenchmarkingAgent | None,
) -> EnvironmentScorecard | None:
    scorecard = arcade.close_scorecard(card_id)
    if agent is not None:
        agent.cleanup(scorecard)
    return scorecard


def _run(args: argparse.Namespace) -> None:
    base_url = args.base_url or os.getenv("PLAP_DEV_BASE_URL")
    api_key = os.getenv("PLAP_DEV_API_KEY")
    model = args.model or os.getenv("PLAP_DEV_MODEL") or "plap-ai/mote"
    if not base_url or not api_key:
        raise SystemExit("PLAP_DEV_BASE_URL and PLAP_DEV_API_KEY are required; start PLAP with `pixi run dev` first.")

    arc_dir = DEFAULT_ARC_DIR.resolve()
    arc_dir.mkdir(parents=True, exist_ok=True)
    recordings_dir = arc_dir / "recordings"
    os.environ["RECORDINGS_DIR"] = str(recordings_dir)
    os.chdir(arc_dir)

    mode = OperationMode.OFFLINE if args.offline else OperationMode.NORMAL
    arcade = Arcade(
        operation_mode=mode,
        environments_dir=str(arc_dir / "environments"),
        recordings_dir=str(recordings_dir),
    )
    card_id = arcade.open_scorecard(tags=["agent", "benchmarkingagent"])
    agent: LocalBenchmarkingAgent | None = None
    scorecard: EnvironmentScorecard | None = None

    try:
        environment = arcade.make(
            args.game,
            scorecard_id=card_id,
            save_recording=True,
            include_frame_data=True,
        )
        if environment is None:
            raise RuntimeError(f"ARC game {args.game!r} is unavailable")

        agent = LocalBenchmarkingAgent(
            card_id=card_id,
            game_id=args.game,
            agent_name="benchmarkingagent",
            ROOT_URL=DEFAULT_ARC_ROOT_URL,
            record=True,
            arc_env=environment,
            config=MODEL_CONFIG_ID,
            local_model_config=_model_config(args, base_url, model),
            action_limit=args.max_actions,
        )
        budget = args.max_actions if args.max_actions is not None else agent.MAX_ACTIONS
        print(f"Running official BenchmarkingAgent locally: game={args.game}, model={model}, action_limit={budget}, scorecard={card_id}")
        print(f"Run recording: {arc_dir / agent.run_dir}")
        agent.main()
    finally:
        scorecard = _close_scorecard(arcade, card_id, agent)

    if scorecard is not None:
        print(scorecard.model_dump_json(indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_environment()
    _run(_parse_args())


if __name__ == "__main__":
    main()
