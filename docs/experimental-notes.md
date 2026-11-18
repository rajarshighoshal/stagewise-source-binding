# Instrument Design and Split Notes

This note records the split locks, checkpoint settings, and two earlier readout iterations behind the `explicit_12` experiment reported in [README.md](../README.md).

## Checkpoints and inference settings

All four stages share the Llama 3.1 tokenizer vocabulary up to Tulu's added chat template and unused `<pad>` entry, and all experiments run on a byte-identical plain-text prompt without chat templates:

1. **Base:** `meta-llama/Llama-3.1-8B`
2. **SFT:** `allenai/Llama-3.1-Tulu-3-8B-SFT`
3. **DPO:** `allenai/Llama-3.1-Tulu-3-8B-DPO`
4. **RLVR:** `allenai/Llama-3.1-Tulu-3-8B`

Every stage runs in `bfloat16` on a single GPU with greedy decoding, explicit `<|end_of_text|>` EOS and batch-padding IDs, and `use_cache=True` during generation (the released Tulu RLVR config defaults to `use_cache=False`, so inheriting checkpoint defaults would confound throughput and generation behavior). Standalone next-token scoring passes discard `past_key_values`.

## Disjoint GSM8K item splits

Items are partitioned deterministically by salted SHA-256 ranking (`seed = 20260821`) in `configs/frame_stage_binding_repair_v2.locked.json` and `configs/frame_stage_binding_recalibration.locked.json`:

- **Pilot (`24` train items):** Compares three single-token mode-choice interfaces (`explicit_ra`, `explicit_12`, `response_mode_field`) across all four stages using choice-token probability mass and `all_reason` vs. `all_direct` control separation, blind to `request_cue` vs. `notes_cue` source effects. Winner: `explicit_12`.
- **Validation (`176` train items):** Evaluates `explicit_12` on instrument checks (`48`-item generation subset, `512`-token cap) and estimates control headroom `H` and the initial source-swap placebo constant `N`.
- **Confirmation (`1,319` test items):** Evaluates `request_cue` and `notes_cue` under both block orders (`request_first`, `notes_first`) on the complete GSM8K test split, with a `64`-item generation audit subset and an independent recomputation check (`frame_stage_binding/repair_recompute.py`).
- **Paraphrase recalibration (`352` train items):** Draws `352` untouched GSM8K training items outside the `200` pilot+validation items (`Base` and `RLVR`, scoring only) to measure fixed-slot directive wording sensitivity without swapping sources.

## Retired readout iterations

Two earlier prompt/parser pairs failed validation checks on the `24`-item pilot before `explicit_12` was frozen:

1. **Bare `R`/`A` sentinel:** Asking Base `Llama-3.1-8B` to emit a bare `R` or `A` produced low candidate-token mass and `4–24%` terminal repetition loops under `1,024`-token continuations, while one `DPO` cell frequently jumped straight to reasoning text without emitting the leading letter.
2. **Repair v1 (`"Write exactly R or A as the next token"` + line-anchored `FINAL:` parser):** Two defects surfaced in raw pilot generations. First, `SFT` obeyed *"as the next token"* literally on `123` rows by emitting the choice token followed immediately by `<|end_of_text|>`. Second, the v1 parser required `FINAL:` at the start of a line after an explicit `REASONING:` header, misclassifying `165` of `267` `RLVR` responses that reasoned normally and placed `FINAL:` mid-line.

Repair v2 (`explicit_12`) replaced the stop-inducing wording with *"Start your response with 1 or 2, then continue in that mode"* and read semantic mode from the last `FINAL:` marker anywhere in the completion (reasoning iff non-whitespace text precedes `FINAL:`), raising post-trained classifiability to `0.95–1.00` without altering the `request_cue` vs. `notes_cue` contrast or scored token position.
