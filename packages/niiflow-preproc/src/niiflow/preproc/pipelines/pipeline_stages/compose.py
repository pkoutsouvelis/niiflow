"""Compose pipeline stages."""

from __future__ import annotations

__all__ = [
    "Compose",
]

from collections.abc import Callable, Sequence
from typing import Any

from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import (
    PipelineStage,
    RuntimeContext,
)


class Compose(PipelineStage):
    """Run a sequence of pipeline stages on a shared runtime context.

    ``Compose`` executes multiple :class:`PipelineStage` instances in order.
    Stages do not pass outputs to each other directly; instead, every stage
    reads from and writes to the same :class:`RuntimeContext`.

    Each contained stage is run normally with :meth:`PipelineStage.run`.
    Therefore, outputs, metadata, saved paths, and completed step ids are
    recorded exactly as they are for ordinary pipeline stages.

    The contained stages are recorded as flat pipeline steps. For example,

    ``Compose([stage_a, stage_b], step_ids=["a", "b"])``

    updates the context as if ``stage_a.run(ctx)`` and ``stage_b.run(ctx)`` had
    been called directly in sequence.

    Args:
        stages: Ordered sequence of child stages.
        step_ids: Optional per-stage ids aligned with ``stages``. When omitted,
            every slot defaults to ``None`` (auto-generated ids at run time).
        params: present for :class:`PipelineStage` but not supported;
            pass them on each child stage.
        save_options: present for :class:`PipelineStage` but not supported;
            pass them on each child stage.
        verbose: Whether informational log messages are emitted.
    """

    def __init__(
        self,
        stages: Sequence[PipelineStage] | None = None,
        *,
        step_ids: Sequence[str | None] | None = None,
        params: dict[str, Any] | None = None,
        save_options: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> None:
        if params:
            raise ValueError(
                f"{type(self).__name__} does not accept `params`; configure each child "
                "stage directly."
            )
        if save_options:
            raise ValueError(
                f"{type(self).__name__} does not accept `save_options`; configure each "
                "child stage directly."
            )
        super().__init__(verbose=verbose)
        self.set_stages(stages, step_ids=step_ids)

    @property
    def stages(self) -> tuple[PipelineStage, ...]:
        return self._stages

    @property
    def step_ids(self) -> tuple[str | None, ...]:
        return self._step_ids

    def set_stages(
        self,
        stages: Sequence[PipelineStage] | None,
        *,
        step_ids: Sequence[str | None] | None = None,
    ) -> None:
        """Replace child stages and step ids."""
        if stages is None:
            normalized_stages: tuple[PipelineStage, ...] = ()
        else:
            if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
                raise TypeError(
                    f"`stages` must be a sequence of PipelineStage instances, "
                    f"got {type(stages).__name__}"
                )
            validated: list[PipelineStage] = []
            for index, item in enumerate(stages):
                if not isinstance(item, PipelineStage):
                    raise TypeError(
                        f"Stage at index {index} must be a PipelineStage, "
                        f"got {type(item).__name__}"
                    )
                validated.append(item)
            normalized_stages = tuple(validated)

        if step_ids is None:
            normalized_step_ids: tuple[str | None, ...] = (None,) * len(
                normalized_stages
            )
        else:
            if not isinstance(step_ids, Sequence) or isinstance(step_ids, (str, bytes)):
                raise TypeError(
                    f"`step_ids` must be a sequence of strings or None, "
                    f"got {type(step_ids).__name__}"
                )
            if len(step_ids) != len(normalized_stages):
                raise ValueError(
                    f"`step_ids` length ({len(step_ids)}) must match "
                    f"`stages` length ({len(normalized_stages)})"
                )
            ids: list[str | None] = []
            for index, step_id in enumerate(step_ids):
                if step_id is None:
                    ids.append(None)
                    continue
                if not isinstance(step_id, str) or not step_id:
                    raise ValueError(
                        f"step id at index {index} must be a non-empty string "
                        f"or None, got {step_id!r}"
                    )
                ids.append(step_id)
            normalized_step_ids = tuple(ids)

        self._stages = normalized_stages
        self._step_ids = normalized_step_ids

    def _resolve_stage_entries(
        self, parent_step_id: str | None
    ) -> list[tuple[str | None, PipelineStage]]:
        """Resolve stage entries as (step_id, stage) tuples."""
        entries: list[tuple[str | None, PipelineStage]] = []
        for index, (stage, step_id) in enumerate(zip(self.stages, self.step_ids)):
            if step_id is not None:
                entries.append((step_id, stage))
            elif parent_step_id is None:
                entries.append((None, stage))
            else:
                entries.append((f"{parent_step_id}.{index}", stage))
        return entries

    def get_index_of_first(
        self, predicate: Callable[[PipelineStage], bool]
    ) -> int | None:
        """Return the index of the first child stage satisfying ``predicate``."""
        for index, stage in enumerate(self.stages):
            if predicate(stage):
                return index
        return None

    def flatten(self) -> Compose:
        """Return a :class:`Compose` with nested compose stages inlined."""
        new_stages: list[PipelineStage] = []
        new_step_ids: list[str | None] = []
        for stage, step_id in zip(self.stages, self.step_ids):
            if type(stage) is Compose:
                flat = stage.flatten()
                new_stages.extend(flat.stages)
                new_step_ids.extend(flat.step_ids)
            else:
                new_stages.append(stage)
                new_step_ids.append(step_id)
        return Compose(new_stages, step_ids=new_step_ids, verbose=self.verbose)

    def __len__(self) -> int:
        """Return the number of stages after :meth:`flatten`."""
        return len(self.flatten().stages)

    def load_param(self, key: str, value: Any) -> Any:
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        raise RuntimeError(
            f"{type(self).__name__} orchestrates child stages via `run`; "
            "`forward` is not used."
        )

    def save_output(self, key: str, value: Any, output_path: Any) -> Any:
        raise RuntimeError(
            f"{type(self).__name__} orchestrates child stages via `run`; "
            "`save_output` is not used."
        )

    def run(
        self,
        ctx: RuntimeContext | None = None,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> RuntimeContext:
        """Execute child stages sequentially and return the updated context.

        The returned context is the same object throughout — artifacts, metadata,
        and ``steps_completed`` accumulate **flat**, as if the children had been
        invoked directly in sequence rather than grouped under :class:`Compose`.

        Auto-generated ids are resolved inside each child's :meth:`~PipelineStage.run`:
        the suffix is ``len(steps_completed)`` at the moment that child starts.
        If ``n`` steps completed before this compose runs, its first
        auto-generated child is ``step_{n:04d}``, the second ``step_{n+1:04d}``,
        and so on.

        When ``ctx.step_id`` is set at :meth:`run` entry, bare stages (``step_ids``
        slot ``None``) receive ``{ctx.step_id}.{index}`` ids.

        Args:
            ctx: runtime context threaded through each child stage.
            start: index of the first child stage to execute (inclusive).
            end: index of the last child stage to execute (exclusive). When
                omitted, all stages from ``start`` are run.
        """
        if ctx is None:
            ctx = RuntimeContext()
        elif not isinstance(ctx, RuntimeContext):
            raise TypeError(
                f"`ctx` must be a RuntimeContext or None, got {type(ctx).__name__}"
            )

        if not isinstance(start, int) or start < 0:
            raise ValueError(f"`start` must be a non-negative int, got {start!r}")
        if end is not None and (not isinstance(end, int) or end < start):
            raise ValueError(f"`end` must be an int >= `start` ({start}), got {end!r}")

        parent_step_id = ctx.step_id
        if parent_step_id is not None and (
            not isinstance(parent_step_id, str) or not parent_step_id
        ):
            raise TypeError(
                f"parent step id must be a non-empty string, got {parent_step_id!r}"
            )

        entries = self._resolve_stage_entries(parent_step_id)
        selected = entries[start:end]

        name = type(self).__name__
        if selected:
            self.log(
                f"[Stage {name}] Running {len(selected)} nested stage(s) "
                f"[{start}:{end if end is not None else len(entries)}]..."
            )

        for step_id, stage in selected:
            ctx.step_id = step_id
            ctx = stage.run(ctx)

        if selected:
            self.log(f"[Stage {name}] Finished nested pipeline.")

        return ctx

    __call__ = run
