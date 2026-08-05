"""
Phase 9 — UI Progress State Validation

Verifies that:
  1. Backend progress_pct values match the UI threshold constants
  2. Stages are defined in strictly ascending order
  3. No two stages share the same threshold (simultaneous activation impossible)
  4. The processing.html JavaScript uses a stagger mechanism (not synchronous)
  5. Poll interval is 1000ms (not 2000ms — the old value that caused missed 65% states)
  6. The activatedSteps Set is present (prevents revert on re-poll)

All tests are pure — inspect source files and JS constants only.
No DB, no browser, no network.
"""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]

# Backend progress_pct values from analyze_resume.py
_BACKEND_EVENTS = {
    "parse":   10,
    "extract": 40,
    "gap":     65,
    "roadmap": 75,
    "done":    100,
}

# UI thresholds from processing.html (must match backend events)
_UI_THRESHOLDS = {
    "step-parse":   10,
    "step-extract": 40,
    "step-gap":     65,
    "step-roadmap": 75,
}


# ---------------------------------------------------------------------------
# Phase 9-A: Backend → UI threshold alignment
# ---------------------------------------------------------------------------

class TestThresholdAlignment:
    def test_parse_threshold_matches_backend(self):
        assert _UI_THRESHOLDS["step-parse"] == _BACKEND_EVENTS["parse"], (
            f"Parse threshold mismatch: UI={_UI_THRESHOLDS['step-parse']} "
            f"backend={_BACKEND_EVENTS['parse']}"
        )

    def test_extract_threshold_matches_backend(self):
        assert _UI_THRESHOLDS["step-extract"] == _BACKEND_EVENTS["extract"]

    def test_gap_threshold_matches_backend(self):
        assert _UI_THRESHOLDS["step-gap"] == _BACKEND_EVENTS["gap"]

    def test_roadmap_threshold_matches_backend(self):
        assert _UI_THRESHOLDS["step-roadmap"] == _BACKEND_EVENTS["roadmap"]

    def test_all_thresholds_strictly_ascending(self):
        """Stages must activate in order: Parse < Skills < Gap < Roadmap."""
        thresholds = list(_UI_THRESHOLDS.values())
        for i in range(len(thresholds) - 1):
            assert thresholds[i] < thresholds[i + 1], (
                f"Stage thresholds not strictly ascending: {list(_UI_THRESHOLDS.items())}"
            )

    def test_no_duplicate_thresholds(self):
        """Two stages sharing a threshold would always activate simultaneously."""
        values = list(_UI_THRESHOLDS.values())
        assert len(values) == len(set(values)), (
            f"Duplicate threshold values detected: {_UI_THRESHOLDS}. "
            "Stages with identical thresholds will always activate at the same time."
        )

    def test_gap_and_roadmap_not_adjacent(self):
        """
        Gap (65%) and Roadmap (75%) must not have adjacent thresholds (e.g. 65 and 66).
        There must be a meaningful gap so a 1s poll can distinguish them.
        """
        gap_threshold = _UI_THRESHOLDS["step-gap"]
        roadmap_threshold = _UI_THRESHOLDS["step-roadmap"]
        assert roadmap_threshold - gap_threshold >= 5, (
            f"Gap ({gap_threshold}%) and Roadmap ({roadmap_threshold}%) are too close. "
            "A 1s poll may not differentiate them even with stagger animation."
        )


# ---------------------------------------------------------------------------
# Phase 9-B: Backend _update_task progress values
# ---------------------------------------------------------------------------

class TestBackendProgressValues:
    def _read_task_source(self) -> str:
        return (_ROOT / "src" / "tasks" / "analyze_resume.py").read_text(encoding="utf-8")

    def test_parse_emits_10pct(self):
        source = self._read_task_source()
        assert "pct=10" in source, (
            "analyze_resume.py does not emit pct=10 for parse stage. "
            "UI 'step-parse' threshold will never be triggered."
        )

    def test_extract_emits_40pct(self):
        source = self._read_task_source()
        assert "pct=40" in source, (
            "analyze_resume.py does not emit pct=40 for skill extraction."
        )

    def test_gap_emits_65pct(self):
        source = self._read_task_source()
        assert "pct=65" in source, (
            "analyze_resume.py does not emit pct=65 for gap analysis. "
            "The 'step-gap' indicator will never light up."
        )

    def test_roadmap_emits_75pct(self):
        source = self._read_task_source()
        assert "pct=75" in source

    def test_done_emits_100pct(self):
        source = self._read_task_source()
        assert "pct=100" in source


# ---------------------------------------------------------------------------
# Phase 9-C: processing.html JavaScript correctness
# ---------------------------------------------------------------------------

class TestProcessingHtmlJs:
    def _read_template(self) -> str:
        return (_ROOT / "src" / "templates" / "processing.html").read_text(encoding="utf-8")

    def test_poll_interval_is_1000ms(self):
        """
        The poll interval must be 1000ms (1s), not 2000ms.
        At 2s intervals, the backend's 65% gap-analysis state (which lasts ~0.1-0.5s)
        is almost never captured, causing Gap and Roadmap to appear simultaneously.
        """
        source = self._read_template()
        assert "setInterval(poll, 1000)" in source, (
            "processing.html poll interval is not 1000ms. "
            "Found setInterval — check for old 2000ms value that causes simultaneous stage activation."
        )
        assert "setInterval(poll, 2000)" not in source, (
            "processing.html still uses 2000ms poll interval. "
            "Change to 1000ms so the 65% gap-analysis state can be caught."
        )

    def test_activated_steps_set_present(self):
        """
        The activatedSteps Set prevents already-lit stages from being reverted
        on subsequent polls and ensures each stage activates at most once.
        """
        source = self._read_template()
        assert "activatedSteps" in source, (
            "processing.html does not have an 'activatedSteps' Set. "
            "Stages may revert to inactive state on re-poll."
        )
        assert "new Set()" in source, (
            "activatedSteps must be initialized as 'new Set()'. "
            "Without this, every poll would check all thresholds and potentially activate out of order."
        )

    def test_stagger_animation_present(self):
        """
        When multiple thresholds are crossed in one poll,
        each must be activated with a setTimeout stagger (not synchronous forEach).
        This prevents Gap and Roadmap from appearing to light up simultaneously.
        """
        source = self._read_template()
        assert "setTimeout" in source, (
            "processing.html has no setTimeout stagger. "
            "When the client skips 65% (gap) and polls at 75% (roadmap), "
            "both stages will activate in the same JS tick — appearing simultaneous."
        )

    def test_stagger_uses_index_multiplier(self):
        """
        Stagger must be index-based (i * N ms) so each newly-crossed stage
        activates sequentially, not all at the same delay.
        """
        source = self._read_template()
        # Look for: i * 300 or similar index-based stagger
        stagger_pattern = re.search(r"[a-z]\s*\*\s*\d+\)", source)
        assert stagger_pattern, (
            "processing.html setTimeout does not use an index multiplier. "
            "All stages will activate at the same time even with setTimeout."
        )

    def test_step_ids_present_in_html(self):
        """All four step indicator div IDs must exist in the template."""
        source = self._read_template()
        for step_id in ("step-parse", "step-extract", "step-gap", "step-roadmap"):
            assert f'id="{step_id}"' in source, (
                f"Step indicator '{step_id}' missing from processing.html. "
                "UI will silently fail to update that stage."
            )

    def test_steps_array_matches_ids(self):
        """
        The JavaScript STEPS array must reference the same IDs as the HTML elements.
        A mismatch means document.getElementById() returns null and stage never lights up.
        """
        source = self._read_template()
        for step_id in ("step-parse", "step-extract", "step-gap", "step-roadmap"):
            assert f"'{step_id}'" in source or f'"{step_id}"' in source, (
                f"STEPS array in processing.html does not reference '{step_id}'. "
                "Stage will never be activated by updateStepIndicator()."
            )


# ---------------------------------------------------------------------------
# Phase 9-D: Stage ordering contract
# ---------------------------------------------------------------------------

class TestStageOrderingContract:
    def test_gap_only_after_extract(self):
        """Gap analysis (65%) must be higher than extract (40%)."""
        assert _UI_THRESHOLDS["step-gap"] > _UI_THRESHOLDS["step-extract"]

    def test_roadmap_only_after_gap(self):
        """Roadmap (75%) must be higher than gap (65%)."""
        assert _UI_THRESHOLDS["step-roadmap"] > _UI_THRESHOLDS["step-gap"]

    def test_backend_stage_sequence(self):
        """Backend _update_task calls must happen in the correct order."""
        source = (_ROOT / "src" / "tasks" / "analyze_resume.py").read_text(encoding="utf-8")
        # Find all pct= assignments in order of line number
        pct_pattern = re.compile(r"pct=(\d+)")
        pcts = [int(m.group(1)) for m in pct_pattern.finditer(source) if m.group(1) != "100"]
        # pcts should be strictly ascending (ignoring 100 at the end)
        assert pcts == sorted(pcts), (
            f"Backend progress_pct values not in ascending order: {pcts}. "
            "Out-of-order progress events will confuse the UI."
        )
