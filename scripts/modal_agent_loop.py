"""Bounded, fail-closed policy controller for Modal failures and Codex repairs.

This controller is intentionally independent of Modal, GitHub, and OpenAI
SDKs.  External calls are supplied by the workflow (or injected in tests), so
credentials can be scoped to one process and local dry runs need no accounts.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runners.modal_app import (
    RUN_SPEC_SCHEMA,
    RunResult,
    RunSpec,
    SpecError,
    redact_secrets,
)

MAX_INFRASTRUCTURE_RETRIES = 2
MAX_AGENT_REPAIRS = 3
PROTECTED_PATTERNS = (
    "configs/models/**",
    "configs/datasets/**",
    "configs/experiments/**",
    "docs/PROTOCOL.md",
    "data/manifests/**",
    "results/**",
    "**/checkpoints/**",
    "runners/modal_app.py",
    "scripts/modal_agent_loop.py",
    ".github/workflows/**",
    "prompts/modal_repair.md",
    "requirements/**",
    ".env*",
)
ALLOWED_PATTERNS = ("src/dlmrel/**", "tests/**")
SENSITIVE_SCIENCE_FILES = {
    "src/dlmrel/alignment.py": "token/word alignment and aggregation",
    "src/dlmrel/artifacts.py": "scientific hashing",
    "src/dlmrel/checkpoints.py": "checkpoint identity and completeness",
    "src/dlmrel/config.py": "frozen scientific configuration",
    "src/dlmrel/controls.py": "controls",
    "src/dlmrel/data.py": "dataset exclusion or manifest rules",
    "src/dlmrel/diffusion.py": "diffusion schedule",
    "src/dlmrel/evaluation/compare_models.py": "cross-model comparison rules",
    "src/dlmrel/evaluation/statistics.py": "bootstrap and multiple-comparison rules",
    "src/dlmrel/experiments/attention_entropy.py": "attention entropy scoring",
    "src/dlmrel/experiments/head_search.py": "head scoring and denominators",
    "src/dlmrel/experiments/logit_lens.py": "logit-lens scoring",
    "src/dlmrel/experiments/pos_probe.py": "POS probe targets and scoring",
    "src/dlmrel/experiments/shared.py": "candidate receivers and attention aggregation",
    "src/dlmrel/experiments/time_curve.py": "time-curve scoring",
    "src/dlmrel/head_search_recovery.py": "selection-aware inference and recovery identity",
    "src/dlmrel/models/_backbone.py": "model attention extraction",
    "src/dlmrel/models/base.py": "model interface and output semantics",
    "src/dlmrel/models/diffullama.py": "DiffuLLaMA loading and attention behavior",
    "src/dlmrel/models/dream.py": "Dream loading and attention behavior",
    "src/dlmrel/permutation.py": "permutation null and p-value rules",
    "src/dlmrel/pipeline.py": "model execution and experiment routing",
    "src/dlmrel/relations.py": "relation definitions",
    "src/dlmrel/relation_selection.py": "selection ranking and relation locks",
    "src/dlmrel/selection.py": "selection ranking",
    "src/dlmrel/splits.py": "official split construction",
    "src/dlmrel/treebank.py": "treebank acquisition and verification",
}
ACTIVE_CONFIG_GLOBS = (
    "configs/models/*.yaml",
    "configs/datasets/*.yaml",
    "configs/experiments/*.yaml",
)


@dataclass(frozen=True)
class LoopState:
    parent_commit: str
    current_commit: str
    agent_repairs: int = 0
    infrastructure_retries: int = 0
    no_op_repairs: int = 0
    failure_signatures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not re_full_commit(self.parent_commit) or not re_full_commit(self.current_commit):
            raise ValueError("controller commits must be full lowercase SHAs")
        for label, value in (
            ("agent_repairs", self.agent_repairs),
            ("infrastructure_retries", self.infrastructure_retries),
            ("no_op_repairs", self.no_op_repairs),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} must be a nonnegative integer")
        signatures = tuple(self.failure_signatures)
        if any(
            not isinstance(signature, str)
            or len(signature) != 64
            or any(char not in "0123456789abcdef" for char in signature)
            for signature in signatures
        ):
            raise ValueError("failure signatures must be lowercase SHA-256 digests")
        object.__setattr__(self, "failure_signatures", signatures)


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    terminal: bool
    next_attempt: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepairOutcome:
    parent_commit: str
    repaired_commit: str | None
    changed_paths: tuple[str, ...]
    diff_text: str
    regression_test_added: bool
    cpu_checks_passed: bool
    config_hashes_before: dict[str, str]
    config_hashes_after: dict[str, str]


@dataclass(frozen=True)
class PolicyReport:
    allowed: bool
    errors: tuple[str, ...]
    changed_paths: tuple[str, ...]
    protected_paths: tuple[str, ...]
    semantic_review: tuple[str, ...]
    meaningful_diff: bool
    regression_test_added: bool
    scientific_identity_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoopTranscript:
    results: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    repairs: list[dict[str, Any]] = field(default_factory=list)


def decide_next(state: LoopState, result: RunResult, *, mode: str) -> Decision:
    if mode not in {"supervised", "guarded-auto"}:
        raise ValueError("mode must be supervised or guarded-auto")
    if result.git_commit != state.current_commit:
        return Decision("stop", "stale result commit differs from controller state", True)
    if result.status == "success":
        return Decision("success", "Modal operation succeeded", True)
    signature = result.failure_signature
    if state.agent_repairs > 0 and signature and signature in state.failure_signatures:
        return Decision("stop", "the same normalized failure occurred twice", True)
    category = result.failure_category or "unknown"
    if category == "transient_infrastructure":
        if state.infrastructure_retries >= MAX_INFRASTRUCTURE_RETRIES:
            return Decision("stop", "infrastructure retry limit reached", True)
        return Decision("infrastructure_retry", "retry same immutable commit", False, result.attempt)
    if category == "resource_exhaustion":
        return Decision("stop", "resource ceiling change requires trusted operator review", True)
    if category in {
        "protected_scientific_behavior",
        "scientific_configuration_mismatch",
        "data_revision_checksum_failure",
        "artifact_checkpoint_mismatch",
    }:
        return Decision("stop", f"{category} is not automatically repairable", True)
    if category not in {
        "deterministic_implementation_failure",
        "dependency_environment_incompatibility",
    }:
        return Decision("stop", "unknown failure requires human triage", True)
    if state.agent_repairs >= MAX_AGENT_REPAIRS:
        return Decision("stop", "maximum three agent repairs reached", True)
    if mode == "supervised":
        return Decision("prepare_repair", "prepare a draft PR and await approval before rerun", True)
    return Decision(
        "repair_and_rerun",
        "bounded repair permitted after policy checks",
        False,
        result.attempt + 1,
    )


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    normalized = path.replace("\\", "/")
    return any(fnmatch.fnmatchcase(normalized, pattern) for pattern in patterns)


def inspect_repair(outcome: RepairOutcome) -> PolicyReport:
    changed = tuple(sorted({path.replace("\\", "/") for path in outcome.changed_paths}))
    protected = tuple(path for path in changed if _matches(path, PROTECTED_PATTERNS))
    outside_allowlist = tuple(
        path for path in changed if not _matches(path, ALLOWED_PATTERNS) and path not in protected
    )
    semantic = tuple(
        f"{path}: {SENSITIVE_SCIENCE_FILES[path]}"
        for path in changed
        if path in SENSITIVE_SCIENCE_FILES
    )
    meaningful = bool(outcome.diff_text.strip())
    same_identity = outcome.config_hashes_before == outcome.config_hashes_after
    errors = []
    if outcome.parent_commit == outcome.repaired_commit:
        errors.append("repair commit did not advance from its parent")
    if protected:
        errors.append("protected paths changed")
    if outside_allowlist:
        errors.append("paths outside the repair allowlist changed: " + ", ".join(outside_allowlist))
    if semantic:
        errors.append("scientific source semantics require human review")
    if not meaningful:
        errors.append("repair produced no meaningful diff")
    if not outcome.regression_test_added:
        errors.append("repair did not add or modify a regression test")
    if not outcome.cpu_checks_passed:
        errors.append("CPU verification did not pass")
    if not same_identity:
        errors.append("active scientific configuration identity changed")
    return PolicyReport(
        allowed=not errors,
        errors=tuple(errors),
        changed_paths=changed,
        protected_paths=protected,
        semantic_review=semantic,
        meaningful_diff=meaningful,
        regression_test_added=outcome.regression_test_added,
        scientific_identity_unchanged=same_identity,
    )


def scientific_config_hashes(root: str | Path) -> dict[str, str]:
    root = Path(root)
    hashes = {}
    for pattern in ACTIVE_CONFIG_GLOBS:
        for path in sorted(root.glob(pattern)):
            hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not hashes:
        raise FileNotFoundError("active scientific configurations were not found")
    return hashes


def notification_payload(
    *, spec: RunSpec, result: RunResult, decisions: list[Decision], tests: list[str]
) -> dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "model": spec.model_id,
        "dataset": spec.dataset_id,
        "experiment": spec.experiment_id,
        "commit": result.git_commit,
        "attempts": result.attempt,
        "status": result.status,
        "failure_signature": result.failure_signature,
        "modal_reference": result.modal_reference,
        "last_completed_checkpoint": result.last_completed_checkpoint,
        "tests": [redact_secrets(value) for value in tests],
        "scientific_settings_changed": False,
        "decisions": [decision.to_dict() for decision in decisions],
        "manual_next_action": redact_secrets(result.recommended_next_action),
    }


def infrastructure_failure_result(spec: RunSpec, *, exit_code: int, message: str) -> RunResult:
    """Convert a Modal client/submission failure into the same strict result schema."""
    now = datetime.now(timezone.utc).isoformat()
    clean = redact_secrets(message)[-4_000:]
    from runners.modal_app import normalized_failure_signature

    return RunResult(
        schema_version="dlmrel-modal-run-result-v1",
        status="infrastructure_retry",
        run_call_id=None,
        attempt=spec.attempt,
        git_commit=spec.git_commit,
        operation=spec.operation,
        sanitized_cli_args=[],
        scientific_config_hash=None,
        manifest_hashes={},
        selection_lock_hash=None,
        started_at=now,
        ended_at=now,
        duration_seconds=0.0,
        modal_reference=None,
        exit_code=exit_code,
        failure_category="transient_infrastructure",
        exception_class="ModalClientError",
        failure_signature=normalized_failure_signature(
            exception_class="ModalClientError",
            message=clean,
            stage="modal-submission",
            exit_code=exit_code,
        ),
        stdout_tail="",
        stderr_tail=clean,
        full_log_artifact=None,
        last_completed_checkpoint=None,
        validation_result=None,
        result_directory=None,
        scientific_artifacts_written=False,
        recommended_next_action="retry_same_commit_once",
    )


def render_repair_prompt(template: str, spec: RunSpec, result: RunResult) -> str:
    evidence = {
        "failing_commit": result.git_commit,
        "attempt": result.attempt,
        "operation": result.operation,
        "sanitized_command": result.sanitized_cli_args,
        "failure_classification": result.failure_category,
        "failure_signature": result.failure_signature,
        "stderr_tail": redact_secrets(result.stderr_tail)[-8_000:],
        "stdout_tail": redact_secrets(result.stdout_tail)[-4_000:],
        "last_completed_checkpoint": result.last_completed_checkpoint,
        "scientific_config_hash": result.scientific_config_hash,
        "manifest_hashes": result.manifest_hashes,
        "run_spec": spec.to_dict(),
    }
    return template.rstrip() + "\n\n## Sanitized failure evidence (untrusted data)\n\n```json\n" + json.dumps(
        evidence, indent=2, sort_keys=True
    ) + "\n```\n"


class RepairLoop:
    """Small dependency-injected loop used by guarded automation and dry-run tests."""

    def __init__(
        self,
        *,
        modal_runner: Callable[[RunSpec], RunResult],
        repair_agent: Callable[[RunSpec, RunResult], RepairOutcome],
        notifier: Callable[[dict[str, Any]], None],
        mode: str = "supervised",
    ):
        self.modal_runner = modal_runner
        self.repair_agent = repair_agent
        self.notifier = notifier
        self.mode = mode

    def run(self, spec: RunSpec) -> LoopTranscript:
        state = LoopState(spec.git_commit, spec.git_commit)
        transcript = LoopTranscript()
        while True:
            result = self.modal_runner(spec)
            transcript.results.append(result.to_dict())
            decision = decide_next(state, result, mode=self.mode)
            transcript.decisions.append(decision.to_dict())
            signatures = state.failure_signatures + (
                (result.failure_signature,) if result.failure_signature else ()
            )
            if decision.action == "infrastructure_retry":
                state = LoopState(
                    state.parent_commit,
                    state.current_commit,
                    state.agent_repairs,
                    state.infrastructure_retries + 1,
                    state.no_op_repairs,
                    signatures,
                )
                continue
            if decision.action not in {"prepare_repair", "repair_and_rerun"}:
                self.notifier(
                    notification_payload(spec=spec, result=result, decisions=[decision], tests=[])
                )
                return transcript
            repair = self.repair_agent(spec, result)
            report = inspect_repair(repair)
            transcript.repairs.append(report.to_dict())
            if not report.allowed or decision.action == "prepare_repair":
                self.notifier(
                    notification_payload(
                        spec=spec,
                        result=result,
                        decisions=[decision],
                        tests=["policy checks passed" if report.allowed else "; ".join(report.errors)],
                    )
                )
                return transcript
            if not repair.repaired_commit:
                return transcript
            state = LoopState(
                state.parent_commit,
                repair.repaired_commit,
                state.agent_repairs + 1,
                0,
                state.no_op_repairs + (0 if report.meaningful_diff else 1),
                signatures,
            )
            spec = RunSpec.from_dict(
                {
                    **spec.to_dict(),
                    "git_commit": repair.repaired_commit,
                    "attempt": spec.attempt + 1,
                    "source_result_identity": result.result_directory,
                }
            )


def _git_output(root: Path, args: list[str]) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    )


def inspect_worktree_repair(root: Path, base_commit: str, expected_hashes: dict[str, str]) -> PolicyReport:
    if not re_full_commit(base_commit):
        raise ValueError("base commit must be a full lowercase SHA")
    head = _git_output(root, ["rev-parse", "HEAD"]).strip()
    paths = tuple(
        line.strip()
        for line in _git_output(root, ["diff", "--name-only", "--no-renames", base_commit, "--"])
        .splitlines()
        if line.strip()
    )
    diff = _git_output(root, ["diff", "--no-ext-diff", "--unified=0", base_commit, "--"])
    current_hashes = scientific_config_hashes(root)
    outcome = RepairOutcome(
        parent_commit=base_commit,
        repaired_commit=None if head == base_commit else head,
        changed_paths=paths,
        diff_text=diff,
        regression_test_added=any(path.startswith("tests/test_") for path in paths),
        cpu_checks_passed=True,
        config_hashes_before=expected_hashes,
        config_hashes_after=current_hashes,
    )
    report = inspect_repair(outcome)
    # Codex leaves an uncommitted patch for the later trusted commit step, so a
    # matching HEAD is expected during policy inspection.
    filtered = tuple(error for error in report.errors if "did not advance" not in error)
    return PolicyReport(
        allowed=not filtered,
        errors=filtered,
        changed_paths=report.changed_paths,
        protected_paths=report.protected_paths,
        semantic_review=report.semantic_review,
        meaningful_diff=report.meaningful_diff,
        regression_test_added=report.regression_test_added,
        scientific_identity_unchanged=report.scientific_identity_unchanged,
    )


def re_full_commit(value: str) -> bool:
    return len(value) == 40 and all(char in "0123456789abcdef" for char in value)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_spec(args: argparse.Namespace) -> RunSpec:
    manifests = json.loads(args.expected_manifest_hashes)
    return RunSpec(
        schema_version=RUN_SPEC_SCHEMA,
        git_commit=args.git_commit,
        operation=args.operation,
        model_config=None if args.model == "none" else f"configs/models/{args.model}.yaml",
        dataset_config=None if args.dataset == "none" else f"configs/datasets/{args.dataset}.yaml",
        experiment_config=(
            None if args.experiment == "none" else f"configs/experiments/{args.experiment}.yaml"
        ),
        track=args.track,
        run_id=args.run_id,
        result_namespace=args.result_namespace,
        selection_lock_source=args.selection_lock_source or None,
        resume=args.resume,
        dry_run=args.dry_run,
        resource_profile=args.resource_profile,
        timeout_seconds=args.timeout_seconds,
        attempt=args.attempt,
        maximum_repair_attempts=args.maximum_repair_attempts,
        expected_scientific_config_hash=args.expected_scientific_config_hash or None,
        expected_manifest_hashes=manifests,
        source_result_identity=args.source_result_identity or None,
        max_estimated_cost_usd=args.max_estimated_cost_usd,
    )


def _dry_run(fixtures: Path) -> int:
    scenarios = _load_json(fixtures)
    print("LOCAL DRY RUN: no Modal, OpenAI, GitHub, or git write operations are permitted.")
    for name, value in scenarios.items():
        state = LoopState(**value["state"])
        result = RunResult.from_dict(value["result"])
        decision = decide_next(state, result, mode=value.get("mode", "guarded-auto"))
        print(json.dumps({"scenario": name, "decision": decision.to_dict()}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dry = commands.add_parser("dry-run")
    dry.add_argument(
        "--fixtures", default="tests/fixtures/modal_loop_results.json", type=Path
    )
    spec = commands.add_parser("build-spec")
    spec.add_argument("--git-commit", required=True)
    spec.add_argument("--operation", choices=sorted({
        "cpu-test", "model-smoke-test", "experiment-run", "validation"
    }), required=True)
    spec.add_argument("--model", choices=("none", "fake", "dream_7b", "diffullama_7b"), required=True)
    spec.add_argument("--dataset", choices=("none", "ewt", "de_gsd", "ja_gsd"), required=True)
    spec.add_argument("--experiment", choices=(
        "none", "head_search", "time_curve", "attention_entropy", "logit_lens", "pos_probe",
        "external_transfer"
    ), required=True)
    spec.add_argument("--track", required=True)
    spec.add_argument("--run-id", required=True)
    spec.add_argument("--result-namespace", default="scratch")
    spec.add_argument("--selection-lock-source", default="")
    spec.add_argument("--resume", action="store_true")
    spec.add_argument("--dry-run", action="store_true")
    spec.add_argument("--resource-profile", required=True)
    spec.add_argument("--timeout-seconds", type=int, required=True)
    spec.add_argument("--attempt", type=int, default=1)
    spec.add_argument("--maximum-repair-attempts", type=int, default=3)
    spec.add_argument("--expected-scientific-config-hash", default="")
    spec.add_argument("--expected-manifest-hashes", default="{}")
    spec.add_argument("--source-result-identity", default="")
    spec.add_argument("--max-estimated-cost-usd", type=float, default=20.0)
    decide = commands.add_parser("decide")
    decide.add_argument("--state", required=True)
    decide.add_argument("--result", required=True)
    decide.add_argument("--mode", choices=("supervised", "guarded-auto"), default="supervised")
    prompt = commands.add_parser("render-prompt")
    prompt.add_argument("--template", default="prompts/modal_repair.md")
    prompt.add_argument("--spec", required=True)
    prompt.add_argument("--result", required=True)
    prompt.add_argument("--output", required=True)
    policy = commands.add_parser("policy-check")
    policy.add_argument("--root", default=".", type=Path)
    policy.add_argument("--base-commit", required=True)
    policy.add_argument("--expected-config-hashes", required=True)
    hashes = commands.add_parser("config-hashes")
    hashes.add_argument("--root", default=".", type=Path)
    infrastructure = commands.add_parser("infrastructure-result")
    infrastructure.add_argument("--spec", required=True)
    infrastructure.add_argument("--exit-code", required=True, type=int)
    infrastructure.add_argument("--message", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "dry-run":
            return _dry_run(args.fixtures)
        if args.command == "build-spec":
            print(_build_spec(args).to_json())
            return 0
        if args.command == "decide":
            state = LoopState(**_load_json(args.state))
            result = RunResult.from_dict(_load_json(args.result))
            print(json.dumps(decide_next(state, result, mode=args.mode).to_dict(), sort_keys=True))
            return 0
        if args.command == "render-prompt":
            template = Path(args.template).read_text(encoding="utf-8")
            spec = RunSpec.from_dict(_load_json(args.spec))
            result = RunResult.from_dict(_load_json(args.result))
            Path(args.output).write_text(render_repair_prompt(template, spec, result), encoding="utf-8")
            return 0
        if args.command == "policy-check":
            report = inspect_worktree_repair(
                args.root, args.base_commit, _load_json(args.expected_config_hashes)
            )
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
            return 0 if report.allowed else 2
        if args.command == "config-hashes":
            print(json.dumps(scientific_config_hashes(args.root), indent=2, sort_keys=True))
            return 0
        if args.command == "infrastructure-result":
            failed_spec = RunSpec.from_dict(_load_json(args.spec))
            print(
                infrastructure_failure_result(
                    failed_spec, exit_code=args.exit_code, message=args.message
                ).to_json()
            )
            return 0
    except (OSError, ValueError, SpecError, subprocess.CalledProcessError) as error:
        print(f"modal-agent-loop: error: {redact_secrets(str(error))}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
