# Hybrid Chatbot TODO

Goal: Friendly assistant that handles both casual chat and school orientation reliably.

## Phase 1 - Baseline and Evaluation

- [ ] Define acceptance targets
  - Routing accuracy >= 90%
  - Chat tone average >= 4/5
  - Orientation tone average >= 4/5
  - Recommendation relevance average >= 4/5
- [ ] Build fixed evaluation set (40 prompts)
  - 15 chat
  - 15 orientation
  - 10 ambiguous/mixed
- [ ] Create repeatable evaluation script
  - Run all prompts
  - Capture intent, response, and quick quality metrics
  - Save results as JSON
- [ ] Run baseline and save report

## Phase 2 - Intent Routing Quality

- [ ] Tune classifier prompt and fallback extraction
- [ ] Validate ambiguous prompt behavior (prefer orientation when unclear)
- [ ] Add minimal keyword guardrail tests for orientation false negatives
- [ ] Re-run routing benchmark and compare vs baseline

## Phase 3 - Conversational Quality

- [ ] Improve chat response style consistency
  - Avoid role artifacts
  - Keep concise and natural
- [ ] Improve orientation rewrite quality
  - No labels/JSON wording
  - Smooth 3-5 sentence answer
- [ ] Add low-quality output rejection + deterministic fallback checks
- [ ] Re-run tone benchmark and compare vs baseline

## Phase 4 - Recommendation Relevance

- [ ] Validate profile-based ranking on known scenarios
  - Rabat + SPC + employability
  - Casablanca + SM + prestige
  - Tight budget + public preference
- [ ] Tune ranking weights and fallback rules only if needed
- [ ] Re-run relevance benchmark and compare vs baseline

## Phase 5 - UI and UX Stability

- [ ] Verify message send paths
  - Send button in composer
  - Enter to send, Shift+Enter newline
  - Send profile only button
- [ ] Ensure evidence panel is optional and non-intrusive
- [ ] Mobile sanity check
- [ ] Clear chat and persistence checks

## Phase 6 - Finalize

- [ ] Run full 40-prompt regression
- [ ] Record final scores and known limitations
- [ ] Freeze prompts/settings for production
