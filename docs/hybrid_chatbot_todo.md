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

## Phase 7 - Profile-First Recommender + Assistant Chat UX

- [ ] Separate UI flows into two clear modes: Recommendation Mode and Chat Mode
- [ ] Make Recommendation Mode profile-first (profile + career/filiere before school constraints)
- [ ] Add backend loader for user_career_profiles by userId
- [ ] Validate user_career_profiles schema fields before scoring
- [ ] Integrate inferredCareers into recommendation intent/domain matching
- [ ] Integrate domainScores into ranking score components
- [ ] Define weighted scoring policy for profile, career, domain, and constraints
- [ ] Apply school constraints after profile/career pre-ranking stage
- [ ] Expose score breakdown in backend response for debugging (internal only)
- [ ] Keep user-facing recommendation text natural and non-technical
- [ ] Add API support for conversation sessions (message after message)
- [ ] Persist and send chat_history on each chat request from UI
- [ ] Improve assistant persona prompts for more conversational replies
- [ ] Add guardrails so chat mode does not force recommendation unless asked
- [ ] Add mode toggle in UI with explicit labels and active state
- [ ] Keep profile-only send as a dedicated action in Recommendation Mode
- [ ] Add quick regression set for profile-first ranking behavior
- [ ] Add quick regression set for multi-turn chat continuity and tone
- [ ] Run end-to-end evaluation and compare against current baseline
