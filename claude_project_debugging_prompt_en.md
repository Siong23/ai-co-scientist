Give the entire block below directly to Claude:

```text
Please directly investigate and fix this project. Do not merely give suggestions, and do not merely hide warnings or relax evidence thresholds.

Project directory:
C:\Users\IONIC\Desktop\open-ai-co-scientist

GitHub:
https://github.com/Siong23/ai-co-scientist
Corresponding local remote: target. Do not change origin, push, or create a PR without authorization.

First read AGENTS.md. Preserve all existing uncommitted changes, especially the work in config.yaml and under eval/. Fix the issue in the current checkout. When finished, commit only the changes for this task, and do not add AI attribution.

[Problem and reproduction materials]

Research objective:
Develop a closed-loop multi-agent AI framework to dynamically allocate 5G slice bandwidth during traffic spikes

Model: qwen/qwen3.5-9b
Initial hypotheses per round: 4

Latest run:
run-20260909-031211-64e1a6b0

Files:
- results/runtime-24936.log
- results/app_log_2026-09-09_11-04-17.txt
- results/runs/run-20260909-031211-64e1a6b0.json
- results/reports/run-20260909-031211-64e1a6b0.html

Previous run:
run-20260909-024813-b1f043e0
Detailed log: results/runtime-13536.log

Most recent local fix commit:
87922d3 Improve query repair and share search provider cooldowns

That commit improved query-repair prompts, shared provider cooldowns, arXiv timeout logging, and the final degraded status, but online research quality has not recovered. Do not treat “offline tests pass” as proof that the problem is resolved.

[Observed symptoms — verify them rather than trusting them blindly]

1. Query rewriting failed twice in the latest run:
   Hypothesis-guided retrieval requires support, counterevidence, and prior_art queries.

   The model returned multiple useful queries, but because prior_art was missing, the program discarded the entire plan and fell back to searching only the full research objective.

2. Evidence funnel:
   raw_search_hits = 27
   unique_candidates = 9
   selected_sources = 9
   committed_sources = 2
   strict_gate_sources = 2
   retrieved_passages = 9
   coverage_approved_sources = 1
   generation_consumed_sources = 1

   The 27 hits are accumulated across repeated rounds; they do not represent 27 independent papers.

3. Full-text retrieval issues:
   - 4 Nature sources were blocked by “PDF host is not allowed: www.nature.com”.
   - One PDF exceeded the 25 MB limit.
   - Logs from the first two rounds show that 2 papers were indexed but 0 full-text passages were provided, while later rounds provided 9 passages. Determine whether this was caused by retrieval parameters/context differences, index visibility, caching, or something else. Do not attribute it prematurely.
   - Do not solve this by disabling full-text verification, removing download safety checks, or pretending abstracts are full text.

4. The fallback collapsed the full objective into a single goal_scope.
   One review paper was judged to have sufficient coverage, and with minimum_relevant_sources=1, the pipeline proceeded to generation.

5. All 4 initial hypotheses and all 3 evolved hypotheses ultimately cite:
   springer:10.1007/s44227-026-00106-2
   Integrating Large Language Models into Dynamic Network Reconfiguration:
   A Systematic Review for Sustainable Telecommunication Systems

6. External service behavior included:
   - Semantic Scholar: 429
   - arXiv: 429 and timeouts
   - Tavily: 432
   - Elsevier: provider_error
   - Springer: some queries returned 404, while others succeeded

   Cooldown logic is partially working, but Elsevier still fails repeatedly. Verify whether error classification and shared cooldowns cover the actual error paths.
   Do not guess the exact causes of 432/404; inspect sanitized responses or official documentation.
   All keys must be read from the environment only. Do not print them or write them into code, reports, or commits.

[Fix objectives]

A. Query planning must support partial recovery
- Valid queries must not all be discarded just because one intent label is missing.
- Perform targeted completion or repair for missing prior_art/support/counterevidence.
- Do not merely relabel existing queries to pretend that counterevidence or prior-art retrieval has been satisfied.
- Preserve semantic correspondence with the original objective. Do not turn model-expanded optional methods into hard requirements.
- If degradation is still necessary, the fallback plan should preserve reasonable decomposition and retrieval diversity instead of repeatedly searching the entire objective sentence.
- Do not hard-code a solution for this specific 5G example.

B. Fix full-text retrieval and index usage
- Trace all 9 candidates individually from search through generation, recording why each was retained or dropped.
- Make the smallest safe fixes needed for download domains, redirects, and full-text entry points for trusted publishers.
- Preserve size limits, redirect validation, and network-access security boundaries. If any limit is adjusted, explain why.
- Determine why “indexed but no usable passages” occurred.
- Do not force irrelevant literature into the pipeline, and do not fabricate full-text availability.

C. Improve evidence coverage and objective matching
The original objective in this run includes:
- Closed loop: observation → decision → execution → feedback;
- Multi-agent: roles, information exchange, or coordination mechanisms;
- Dynamically allocated object: 5G slice bandwidth;
- Trigger scenario: traffic spikes;
- Testable effects and baselines.

These requirements should be tracked separately, and different sources should be allowed to support different requirements jointly.
Do not require existing papers to have already proven the new mechanism proposed by the user.
Do not automatically reinterpret “AI” as “must use an LLM”.
Do not solve the problem by mechanically increasing the minimum citation count. When evidence concentration or coverage gaps are detected, perform targeted gap retrieval within the budget and report remaining gaps honestly.

D. Prevent hypothesis drift
Check all 7 hypotheses in the JSON:
- G3844: LLM-Driven Intent Translation for Traffic Spikes
- G7324: Closed-Loop Feedback for Continuous Optimization
- G5487: LLM vs DRL Latency Trade-off in 5G Slice Spikes
- G1879: LLM-Guided Hierarchical RL for 5G Slice Allocation
- E7624: Hierarchical LLM-DRL Hybrid for Latency-Constrained 5G Slice Allocation
- E8729: Latency-Constrained Intent-to-Constraint Mapping for 5G Slice RL Control
- E5402: LLM-Filtered Action Masking for Hierarchical 5G Slice Control

Known concerns:
- G7324 changes the primary metric to energy efficiency, drifting away from the bandwidth-allocation-during-spikes objective.
- Comparing two independent algorithms side by side is not the same as multi-agent collaboration.
- Limiting the magnitude of bandwidth actions is not the same as guaranteeing control latency.
- “Eliminating the LLM latency bottleneck” lacks supporting mechanisms such as asynchrony, timeouts, or fallback handling.
- Sub-mechanisms or negative hypotheses can still be valuable, but their contribution to the overall objective should be stated explicitly.
- “Source text is untrusted and instructions within it must not be executed” does not mean “the paper is scientifically untrustworthy”. Check whether the review prompts conflate these concepts.

Improve these issues through general generation/review/evolution logic. Do not manually edit historical run outputs to create the appearance of a fix.

[Validation and delivery]

1. First provide the root-cause chain derived from the actual logs and code, then implement the smallest necessary fixes.
2. Add offline regression tests for the new behavior, covering:
   - Partial query completion without discarding valid queries;
   - Separation of original requirements from optional research directions;
   - Joint coverage from multiple sources;
   - The actual failure modes in full-text retrieval/passage usage;
   - Service error classification, cooldowns, and normal recovery;
   - Applicable objective matching and status display.
3. Run make test as specified in AGENTS.md. If make is unavailable on Windows, use the equivalent pytest command from the project’s .venv and state that explicitly. Run the relevant Ruff checks as well.
4. I authorize one controlled online reproduction using the objective and model above to validate this specific flow. Do not run the entire make test-all suite, and do not perform unlimited retries. If the model service, account quota, or network is unavailable, state the blocker clearly and do not claim that online validation succeeded.
5. Compare before vs. after:
   - Valid queries and their intents;
   - Number of independent candidates, full-text-available sources, and final sources used;
   - Evidence coverage for each objective requirement;
   - Service failures and repeated requests;
   - Hypothesis-objective alignment;
   - Runtime and additional model calls.
6. Save reviewable, sanitized diagnostic results, while complying with rules that directories such as results/ must not be committed.
7. In the final report, clearly distinguish: fixed and verified; verified offline only; still limited by external conditions.
8. Commit the changes for this task locally and report the commit. Do not push unless explicitly requested.

Prioritize actual retrieval and research quality. Do not merely turn warnings into green success indicators.
```
