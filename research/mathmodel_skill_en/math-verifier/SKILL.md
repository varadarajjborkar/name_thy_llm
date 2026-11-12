---
name: math-verifier
description: Use when a modeling plan for a CUMCM-style (China Undergraduate Mathematical Contest in Modeling) task needs an independent check of formulas, units, dimensions, constraints, boundary cases, feasibility wording, assumptions, or paper-ready mathematical conclusions before code, figures, the abstract, or final submission.
---

# Mathematical Verification

## Purpose

Verify the math before it becomes paper language. Treat this skill as an independent hard gate for dimensional consistency, formula back-substitution, boundary behavior, feasibility claims, and assumption loss.

Core rule: do not solve a new problem and do not invent missing model parts. Verify only what has already been written, computed, or claimed, against the problem statement, the modeling handoff, the symbol table, and the evidence artifacts.

Top-prize candidate bar: mathematical verification must make sure that fit to the problem, modeling insight, credible evidence, reproducibility, and clear boundaries are backed by formulas, units, constraints, and boundary cases. This is a submission-quality bar, not a promise of a prize. A good-looking conclusion that cannot be back-substituted, whose units do not close, or whose boundaries cannot be explained, cannot become paper-ready.

Hard gate: if any unit, domain, constraint, boundary condition, formula back-substitution, or feasibility wording is unchecked, inconsistent, or unknown, the model conclusion cannot become paper-ready.

Return rule: this skill returns its findings only to `math-hub`. It may suggest a repair surface, but it must not route to, dispatch, or call another skill directly.

## Required Inputs

Collect the smallest available set:

- current problem/subquestion and required deliverables;
- `problem_brief.md`, `model_handoff.md`, `assumption_log.csv`, and `symbol_table.csv` when available;
- formulas, constraints, objective, parameter values, units, and domains;
- result tables, `result_registry.csv`, `run_record.csv`, and `claim_ledger.csv` for computed claims;
- official or user-provided tolerances, hard constraints, and feasibility definitions.

If a needed source is missing, mark the affected check `blocked` or `diagnostic-only`; do not fill the gap from memory.

## Verification Checks

Run only checks relevant to the active claim or model route:

| Check type | What to verify |
| --- | --- |
| `dimensional` | Every term in an equation has compatible dimensions and declared units. |
| `domain` | Variables, indices, sets, binary/slack roles, and parameter ranges match the problem and code handoff. |
| `constraint` | Hard constraints are represented, measurable, and not silently softened. |
| `formula back-substitution` | Substitute baseline, boundary, or reported values into formulas to catch algebra/sign/unit errors. |
| `boundary` | Extreme, zero, saturated, or limiting cases behave in the expected direction. |
| `conservation` | Flow, mass, count, probability, budget, or balance laws close within tolerance. |
| `feasibility wording` | Claims use strict feasible, relaxed diagnostic, candidate, not identifiable, or unavoidable gap correctly. |
| `assumption loss` | Simplifications name what realism, constraint, or uncertainty they remove. |
| `result schema` | The table/figure needed to prove the claim can hold all required units, scenarios, baselines, and tolerances. |

Do not reward complexity. A simple model with closed units and checked boundaries is preferred over a complex model that hides a feasibility gap.

## Artifact Contract

When maintaining `math_verification.csv`, use:

```csv
check_id,subquestion,artifact,location,claim_id,check_type,input_ref,expected_relation,observed,status,severity,minimum_fix,owner
```

Statuses:

- `pass`: check is directly satisfied by cited evidence.
- `blocked`: required source, unit, formula, constraint, or tolerance is missing.
- `fail`: evidence contradicts the model or claim.
- `diagnostic-only`: check reduces uncertainty but is not enough for paper-ready use.

Severity:

- `P0`: contradiction, fabricated value, missing required deliverable, or rule-breaking feasibility wording.
- `P1`: unit, constraint, boundary, reproducibility, or validation gap blocking paper-ready status.
- `P2`: weak assumption explanation, missing stress case, or fragile boundary claim.
- `P3`: wording, notation, or presentation cleanup after the math is sound.

## Output Contract

Return:

```text
Verification lock:
Checked artifacts:
Hard check results:
P0/P1 blockers:
Claims allowed as paper-ready:
Claims downgraded to candidate/diagnostic:
Assumption and boundary losses:
Math verification rows:
Smallest repair order:
Return to hub: math-hub
Recommended repair surface:
```

`Recommended repair surface` may name `model`, `code`, `table`, `figure`, `abstract`, `latex`, or `hub QC`, but the handoff always returns to `math-hub`.

## Red Lines

Stop and mark a blocker when:

- a symbol has no unit, domain, or first-use definition;
- two terms are added or compared with incompatible dimensions;
- a hard constraint is absent from the model, code, or feasibility check;
- a reported result cannot be reproduced by formula back-substitution or a cited run;
- a relaxed, violated, or candidate solution is described as strict feasible;
- a boundary case contradicts the claimed mechanism or trend;
- an assumption removes a real constraint without naming the loss;
- verification starts designing a new route instead of checking the current one.

Do not hide failed checks in prose. Downgrade the claim or return the missing evidence to `math-hub`.
