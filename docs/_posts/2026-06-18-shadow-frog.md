---
layout: blog-post
title: "Shadow-Frog: Coding Agents that Dream and Discover"
date: 2026-06-18
author: "MSR Montréal Froggy Team"
reading_time: 20
tags: ["Coding Agents", "Skills", "Software Engineering", "Copilot CLI"]
description: "Shadow-Frog turns idle coding-agent time into autonomous discovery loops, building a shadow knowledge base for any codebase. We evaluate it across retrieval, bug hunting, bug fixing, and feature ideation."
github_url: "https://github.com/microsoft/ShadowFrog"
---

<style>
  @media (max-width: 768px) {
    .post-figure iframe {
      max-height: 70vh !important;
    }
    .post-figure iframe[title*="Three-panel"] {
      height: 680px !important;
      max-height: none !important;
    }
  }
</style>

<figure class="post-figure" markdown="0">
  <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:10px;padding:6px;border:1px solid #2a3760;box-shadow:0 2px 12px rgba(26,26,46,0.18);">
    <iframe src="{{ 'figures/shadow-frog/fig0-hero-sunburst.html' | relative_url }}?v=2"
            title="xarray dream lineage sunburst (n = 115 dreams)"
            style="width:100%;aspect-ratio: 1 / 1.02;border:0;border-radius:7px;display:block;"
            loading="lazy"></iframe>
  </div>
  <figcaption style="text-align:left;font-size:0.92em;color:#555;margin-top:10px;line-height:1.5;">
    <strong>Figure 0 &mdash; A dream campaign in the wild.</strong> One repo&rsquo;s full lineage sunburst (pydata/xarray, <em>n</em> = 115 dreams). Each slice is one autonomous dream; rings move outward as later dreams branch off earlier ones; color marks the <em>kind of work</em> the dream explored (a taxonomy we use again later). We unpack what &ldquo;dreams&rdquo; and &ldquo;compounding&rdquo; mean in the sections that follow.
  </figcaption>
</figure>

## TL;DR
Shadow-Frog is an agentic discovery system that builds a codebase memory for coding agents through active exploration rather than passive recording. Shadow-Frog turns idle coding-agent time into autonomous **discovery loops**. In discovery sessions, agents run experiments on underexplored parts of the codebase and capture learnings in a shadow knowledge base. We find Shadow-Frog agents are better than baselines at retrieving knowledge, detecting bugs, and ideating new features. Unlike other offline memory systems, Shadow-Frog doesn't passively consolidate memories of past tasks. Agents actively learn by doing: acquiring tacit knowledge that human coders only attain by grappling hands-on with a codebase.

When you kick off a discovery session (typically before stepping away from your IDE), the agent doesn't wait for a task: it imagines experiments against your codebase, runs them on isolated git branches, and distills what it learns into a **shadow knowledge base**: a `.shadow/` directory that mirrors the source tree. Every file has a sibling markdown file -- a shadow file -- which captures what agents have learned about how the code in the corresponding file actually behaves. This shadow file structure makes it quicker and easier for agents in future sessions to find relevant discoveries compared to sifting through a flat file. Every subsequent interaction benefits from what the shadow already knows.

<figure class="post-figure" markdown="0">
  <iframe src="{{ 'figures/shadow-frog/fig-tldr.html' | relative_url }}?v=3"
          title="Three-panel results summary: 97.6% recall at budget 8, 88%/22% blind bug localization on 50 real bugs, +25.4 percentage points across 20 repos (wins 15/20)"
          style="width:100%;height:260px;border:0;border-radius:8px;display:block;"
          loading="lazy"></iframe>
</figure>

We evaluate the system across five studies (retrieval, blind bug hunting on synthetic and real bugs, feature ideation, bug fixing), each against a matched baseline (same model, same repo access, no shadow) where applicable. Three headline results:

- **① Shadow recall.** Is a structured shadow actually easier for agents to navigate than a flat file? Shadow-Frog's per-file layout (one shadow file per source file) reaches **97.6% retrieval accuracy** in just 8 tool calls (file reads or searches the agent is allowed to make), across 69,342 runs on django and fastapi. Structure matters: a flat shadow (all content in one file) scores only 36.2% at the same budget.
- **② Blind bug hunting.** Does idle-time exploration find real problems in the code? With no problem statement and no hints, the agent finds bugs purely from idle-time exploration. On 20 repos × 100 synthetic bugs (SWE-Smith), Shadow-Frog beats a baseline by **+25.4 percentage points** (71.5% vs 46.0%), winning 15 of 20 repos. On 50 real bugs from SWE-Bench Verified, it flags the **correct module in 88% of cases and the exact buggy function in 22%**.
- **③ Feature ideation.** Can compounded knowledge help an agent anticipate what to build? Dream campaigns on 8 open-source repos (pinned at January 2023) anticipate features that maintainers actually released afterward: **+10.4 percentage points** alignment over a no-shadow baseline, and a blind quality judge scores Shadow-Frog ideas **+0.40 higher on Insight** (1–5 scale, all 8 repos).

Full numbers, per-task spotlights, and ablations in the Evaluations section below.

## Why we built it

Modern agent memory mostly preserves what the agent has already seen during interaction. But mature codebases hold a different class of knowledge: tacit knowledge that only emerges from *running* things. Which refactor will silently break downstream callers? Where does documentation quietly disagree with implementation? Which invariants does the test suite never actually exercise? This kind of insight is difficult to find in docs and equally difficult to surface with passive memory systems.

We wanted a system that could autonomously gather exactly that kind of knowledge during the agent's idle time: anticipating what a user might need before they ask. During idle time (coffee, lunch, overnight, weekends), Shadow-Frog can *dream* on candidate tasks against the repository (a new feature, a refactor, a security audit, a performance pass) and carries them out in earnest, batch after batch. By analyzing its own attempts, the agent learns things about the codebase that could not have been derived from reading it.

In summary, first, we **imitate how engineers themselves come to know a codebase**: by writing code and running it, not by re-reading the source one more time. Second, we treat the knowledge base as something to be **actively discovered, not passively recorded**: the agent goes looking for what it doesn't know, instead of waiting for material to drift past. And third, the shadow is a home for **knowledge that's hard to surface by reading the source alone**: behavioral facts that only show up when someone actually tries something (implements a new feature, fuzzes an input, exploits a code path).

## The shadow memory substrate

The shadow is a `.shadow/` folder that mirrors the source tree. Every code file has a sibling markdown that holds what the agent has learned about it; insights that span multiple files live in `.shadow/_cross/`, referenced from both sides at the symbol level (`file::function`). **The index is the repo**: there is no vector store, no embedding database, no separate retrieval service. Any agent that can navigate the codebase already knows where the shadow for a given file lives: if you are editing `src/parser.py`, you check `.shadow/src/parser.py.md`.

Discoveries carry their provenance: `exploration` (the agent ran experiments), `user` (the human stated it in conversation), or `interaction` (it emerged during collaborative debugging). User-sourced knowledge is the highest-trust class and is captured with the same fidelity as dream-derived discoveries. Actionable findings can additionally carry one or more of five labels (`bug`, `security`, `performance`, `feature-gap`, `tech-debt`), and a `preToolUse` hook (an automatic trigger that fires before the agent edits a file) inlines the `bug` and `security` ones whenever the agent is about to modify the surrounding code.

Two consequences follow from this design. First, **any agent that can run `cat` and `grep` consumes the shadow out of the box**: no embedding pipeline, no special client, no schema migration. Second, **updates are sparse**: when source files change, `/shadow-frog-update` diffs against the last-known commit and walks only the affected paths, refreshing structural sections and re-checking the discoveries anchored there. The shadow grows and ages with the repo at the granularity of the git diff.

<figure class="post-figure" markdown="0">
  <iframe src="{{ 'figures/shadow-frog/fig1-shadow-tree.html' | relative_url }}?v=4"
          title="Real example from examples/coupon-demo: the source tree (cart.py, inventory.py, test_cart.py) on the left and its sibling .shadow/ mirror on the right. A snippet of cart.py shows the missing .upper() call; the matching shadow markdown captures the bug as a discovery. Below, the _cross/coupon-case-normalization-mismatch.md file shows how one cross-cutting discovery anchors to symbols in both source files via file::symbol references."
          style="width:100%;height:730px;border:0;border-radius:5px;display:block;"
          loading="lazy"></iframe>
  <figcaption style="text-align:left;font-size:0.92em;color:#555;margin-top:10px;line-height:1.5;">
    <strong>Figure 1 &mdash; Shadow mirrors the source tree.</strong> Every source file pairs with a sibling markdown in <code>.shadow/</code>; cross-cutting discoveries live in <code>.shadow/_cross/</code> and are anchored to specific <code>file::symbol</code> locations on both sides. A reader of any one file can still find what we&rsquo;ve learned about how it interacts with the rest of the codebase.
  </figcaption>
</figure>

## Dream: the discovery loop

Every dream is an **experiment**: the agent picks a task it thinks would be useful, opens a parallel git branch, implements it for real, and runs whatever scripts and tests it needs. Whether the *code* gets merged is almost beside the point (most of it doesn't, by design). The real output is what the agent learns about the codebase along the way: concrete behavioral facts that surface only when you actually run code.

### Imagining

The agent imagines candidate tasks targeting under-explored parts of the codebase: a new feature, a refactor, a security audit, a bug hunt, a performance pass. Candidates are spread across six investigation categories and weighted toward files the shadow has barely touched, so the agent doesn't keep poking at code it already understands. Instead of waiting to be asked, the agent surfaces hypotheses about what the codebase will need next, and uses each one as a pretext to probe somewhere it hasn't probed before.

### Executing on real branches

Each dream spawns a worker that opens a parallel git branch, picks up its task, and works through it: reading and writing files, running scripts, calling tests. Every commit lands on a dedicated `dream/<namespace>/<id>` branch. Dream results are **live, runnable code on real branches**, not patches sitting in a file. A reviewer can check one out, run it, see exactly how the agent reached its conclusions.

### Compounding dreams that build on dreams

Today's coding-agent sessions typically wrap up the moment the agent declares success (task done, test green, PR drafted), which keeps each session short and self-contained. The flip side is *shallow* development: every new session restarts from `main` with no memory of where the last one was heading, and investigations that would need more than one sitting rarely happen. Compounding dreams are designed to break this ceiling. Because dream branches persist, a later dream can branch from an earlier dream branch instead of from `main`, e.g., a dream that prototypes a retry layer becomes the starting point for a dream that adds timeout handling, which becomes the starting point for a dream that hardens the entire HTTP layer. Each dream archives a short structured report under `.shadow/_dreams/<id>/` containing base commit, branch name, parent branch, category, and a `useful`/`dead_end` verdict, alongside a manifest of discoveries that a reconciliation script merges into the shadow on `main`. Future dream sessions read past reports first, avoiding dead ends and continuing partially-useful work. Knowledge compounds *across* sessions, not just within one.

<figure class="post-figure" markdown="0">
  <img src="{{ 'figures/shadow-frog/fig2-discovery-loop.svg' | relative_url }}?v=3"
       alt="The discovery loop. Four steps: imagine a task, run it on a dream branch, distill a discovery, merge it to the shadow, then return to imagine. An inset shows three dream branches forking from each other to compound knowledge."
       style="width:100%;max-width:820px;display:block;margin:0 auto;"/>
  <figcaption style="text-align:left;font-size:0.92em;color:#555;margin-top:10px;line-height:1.5;">
    <strong>Figure 2 &mdash; The discovery loop.</strong> Every idle moment starts the cycle on the left: the agent imagines a candidate task, runs it on a dedicated dream branch, distills the result into a one-line discovery, and merges that discovery back into the shadow. The inset on the right shows the compounding mechanism: later dreams branch off earlier dream branches (not <code>main</code>), so each one inherits its parent&rsquo;s code and shadow. The rings in the dream campaign figure at the top of the post are exactly this branching, viewed from above.
  </figcaption>
</figure>

### Distilling discoveries

A dream's experiment code might never be merged into the main repository (most aren't, and that's by design). Nevertheless, the discoveries made by those dreams are the important part: each is a one-sentence behavioral fact, anchored at the right symbol, labeled with its source and tagged `verified` (confirmed by actually running code), `uncertain` (plausible but unconfirmed), or `refuted` (probed and found wrong). All three statuses get merged into the shadow on `main` with their tags intact, so future sessions act on `verified` claims, treat `uncertain` ones as leads, and avoid re-walking `refuted` ground. The dream's archived report, manifest, and patch diff under `.shadow/_dreams/<id>/` are the durable evidence, in case we later want to know *how* we know.

## Evaluations

Unless noted otherwise, all experiments use **Claude Opus 4.6** as the backbone LLM, running inside the **Copilot CLI** agent harness. Where we measure cross-LLM generalization (the robustness re-scoring in §4), we additionally run Opus 4.7 and GPT-5.5 as independent judges.

### 1 · Needle in a shadow haystack

Before measuring what the shadow enables, we verify the agent can read it back. We plant one-sentence synthetic facts ("needles") inside shadow files across the repository, then ask the agent a question whose answer requires those needles. Here is a real example from the django evaluation:

<div style="background: linear-gradient(135deg, #f0fdf4, #f8fafc); border-radius: 10px; padding: 20px 24px; border: 1px solid #d1d5db; margin: 16px 0; font-size: 0.92em; line-height: 1.6; overflow-wrap: break-word; word-break: break-word;">
  <div style="margin-bottom: 14px;">
    <span style="display: inline-block; background: #059669; color: white; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.5px;">Needle</span>
    <span style="color: #6b7280; font-size: 0.85em; margin-left: 8px; word-break: break-all;">.shadow/django/db/models/query_utils.py.md &rarr; Q</span>
  </div>
  <div style="color: #1e293b; font-style: italic; padding-left: 12px; border-left: 3px solid #10b981;">
    kwargs are sorted before being stored as children, so Q(a=1, b=2) and Q(b=2, a=1) produce identical objects.
  </div>
  <div style="margin-top: 16px; margin-bottom: 10px;">
    <span style="display: inline-block; background: #4f46e5; color: white; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.5px;">Prompt</span>
    <span style="color: #6b7280; font-size: 0.85em; margin-left: 8px;">hint given: django/db/models/query_utils.py::Q</span>
  </div>
  <div style="color: #1e293b; font-style: italic; padding-left: 12px; border-left: 3px solid #6366f1;">
    You are about to make changes to Q. Enumerate every prior project knowledge item relevant to making changes here safely.
  </div>
</div>

We test three systems on tasks like this: **Shadow-Frog** (our mirrored layout, one shadow file per source file), **flat shadow** (all discoveries dumped into one giant file), and **no shadow** (same model, no knowledge base). Each system gets the same file-path hint and a budget of 8 tool calls.

Across **69,342 runs** sweeping django + fastapi, 5 shadow sizes (50 → 35k needles), 7 tool-call budgets (1 → unlimited), and 2 task framings, Shadow-Frog reaches **97.6% recall** in the realistic configuration (django, a ~2,500-needle shadow, file path given, **budget of 8 tool calls**), versus a flat shadow at **36.2%** and no-shadow at **12.4%**.

On the example above, the contrast is even starker:

<div style="background: #f8fafc; border-radius: 8px; padding: 16px 20px; border: 1px solid #e5e7eb; margin: 16px 0; font-size: 0.9em; line-height: 1.7; color: #374151; overflow-wrap: break-word; word-break: break-word;">
  <strong>Shadow-Frog:</strong> glob → view the 20-line file → answer. <span style="color: #059669; font-weight: 600;">4 calls, recall 100%.</span><br>
  <strong>Flat shadow:</strong> 7 calls orienting inside a 9,301-line monolith, found the needles on call #8, ran out of budget before answering. <span style="color: #dc2626; font-weight: 600;">Recall 0%.</span><br>
  <strong>No shadow:</strong> grepped the source for <code>class Q</code>, found the definition, but planted facts exist only in the shadow. <span style="color: #dc2626; font-weight: 600;">Recall 0%.</span>
</div>

"File path given" is the realistic posture, not a contrived leg-up. As we noted up top, *the index is the repo*, so whenever an agent is reading a source file it already knows where the matching shadow lives, with no retrieval lookup in between. The per-file layout makes this mechanically derivable (`query_utils.py` → `.shadow/.../query_utils.py.md`), so the agent reaches the right 20-line file in one glob. A flat dump forces a search through hundreds of symbol headings, burning the budget on orientation instead of retrieval. At unlimited budget the flat shadow catches up, but at realistic budgets the **structure** of the shadow matters as much as its content. The story repeats across the must-search row too: every cell with a real shadow size and a realistic budget shows the same shape.

<figure class="post-figure" markdown="0">
  <div style="background:#f8fafc;border-radius:8px;padding:8px;border:1px solid #e5e7eb;">
    <iframe src="{{ 'figures/shadow-frog/fig3-recall.html' | relative_url }}?v=3"
            title="Recall sweep on django: ten panels (2 task framings × 5 shadow sizes), each plotting recall vs tool-call budget for three conditions (no shadow, flat shadow, Shadow-Frog). The headline cell (file path given × 2,500-needle shadow) is highlighted; at a budget of just 8 tool calls, Shadow-Frog reaches 97.6% while flat shadow is at 36.2% and no-shadow is at 12.4%."
            style="width:100%;height:540px;border:0;border-radius:5px;display:block;"
            loading="lazy"></iframe>
  </div>
  <figcaption style="text-align:left;font-size:0.92em;color:#555;margin-top:10px;line-height:1.5;">
    <strong>Figure 3 &mdash; Recall on django: Shadow-Frog vs flat vs no-shadow.</strong> <em>Rows:</em> did the agent receive the file path, or did it have to search? <em>Columns:</em> shadow size (50 &rarr; 35k needles). <em>x-axis:</em> tool-call budget (1 &rarr; &infin;). The headline cell (highlighted) is the realistic case: file path given, 2,500-needle shadow. At a budget of just <strong>8 tool calls</strong>, Shadow-Frog already reaches <strong>97.6%</strong> while flat shadow is at <strong>36.2%</strong> and no-shadow is at <strong>12.4%</strong>. The flat&rarr;Shadow-Frog gap dissolves only at much higher budgets. <em>Hover any point for exact recall; toggle the confidence-interval checkbox.</em>
  </figcaption>
</figure>

### 2 · Blind bug hunting on synthetic bugs

The dream knowledge base is built with no knowledge of what bugs exist. Once the shadow is complete, we point the agent at the repo and ask "what's wrong?", with no problem statement and no hints. We call a bug "found" when an LLM judge rules the agent's report sufficient for a developer to fix the bug from the report alone (the "strict tier"). On 20 repos × 100 synthetic bugs each (SWE-Smith injects non-conflicting bugs simultaneously into a single repo, with all tests removed), with a matched no-shadow baseline swept across 10 budget levels (number of independent sub-agents allocated per task, from 12 to 120), Shadow-Frog beats baseline by **+25.4 percentage points (71.5% vs 46.0%)**, holds the lead at every budget once past the smallest, and wins on **15 of 20 repos**. The largest deltas are on architecturally complex codebases (deepdiff +72 percentage points, pypika +67, astroid +63), where structured exploration of system invariants pays off most.

<figure class="post-figure" markdown="0">
  <div style="background:#f8fafc;border-radius:8px;padding:8px;border:1px solid #e5e7eb;">
    <iframe src="{{ 'figures/shadow-frog/fig4-swesmith.html' | relative_url }}?v=2"
            title="SWE-Smith budget curves (3 panels: Level 1 file match, Level 2 function match, LLM judge verdict; baseline vs Shadow-Frog across 10 budget levels)"
            style="width:100%;height:430px;border:0;border-radius:5px;display:block;"
            loading="lazy"></iframe>
  </div>
  <figcaption style="text-align:left;font-size:0.92em;color:#555;margin-top:10px;line-height:1.5;">
    <strong>Figure 4 &mdash; SWE-Smith: knowledge compounds with budget.</strong> Three increasingly strict bug-identification metrics across 10 budget levels (number of sub-agents per task), averaged over 20 repos &times; 100 injected bugs each. <strong>Level 1</strong> (rule-based): any finding mentions the buggy file. <strong>Level 2</strong> (rule-based): any finding mentions the buggy function. <strong>Judge</strong> (LLM-based): an LLM judge rules the report sufficient to fix the bug. Shadow-Frog leads on the strict-tier Judge bar at every budget, peaking at <strong>+25.4 percentage points</strong> (71.5% vs 46.0%). <em>Hover any point for exact values; toggle the confidence-interval checkbox.</em>
  </figcaption>
</figure>

### 3 · Blind bug hunting on real-world bugs

On 50 real bugs from SWE-Bench Verified, with no problem statement provided, the dream knowledge base flags **the correct module for 88% of bugs and the exact buggy function for 22%** (at the same strict-tier bar defined in §2). A simpler deterministic check (does the bug report name the right function?) hits 84%. This study has only one condition (the shadow agent), because a no-shadow agent has no signal that a specific bug exists in a multi-thousand-file codebase. The natural comparator is "what would we have found without the dream campaign?" (§2 already showed the matched-baseline result at scale on synthetic bugs; here we test on real ones). Two findings surprised us. First, **dream volume barely predicts success**: the correlation between per-task dream count and bug-found rate is effectively zero. A 407-dream task on scikit-learn never reached the strict tier, while a 95-dream task on pytest did. Targeted experiments beat brute-force volume by a wide margin. Second, **dream-on-dream compounding works in clear cases**: five tasks were demonstrably upgraded by later dream sessions building on earlier ones (two xarray tasks reached the strict tier this way). But a substantial share of compounding sessions still revisit prior ground without producing new discoveries. Getting child dreams to reliably *extend* their parents rather than re-walk them is what we are actively pushing on.

### 4 · Feature ideation: anticipating what to build next

Can the shadow help an agent imagine what to build next? We pin 8 open-source repos at January 2023 and build a shadow for each via dream campaigns. Then two agents each generate 50 feature ideas for the same repo: Shadow-Frog reads the shadow; the baseline has identical repo access but no shadow. Neither is told what maintainers actually released in the months after January 2023.

**Alignment to released features.** We collect the 910 features maintainers actually merged after January 2023 and check overlap. Shadow-Frog produces **+10.4 percentage points** more clearly-related ideas on a single generation (28.2% vs 17.8%), winning significantly on 5 of 8 repos. Shadow-Frog also uniquely proposes 166 released features that baseline never reaches (baseline uniquely proposes 71).

**Blind quality judging.** 3,310 ideas scored 1–5 by three independent LLM judges (Opus 4.6, Opus 4.7, GPT-5.5) on four dimensions. Shadow-Frog wins on **Insight (+0.40)** and **User Impact (+0.24)** across all 8 repos; it loses on Groundedness (−0.23) and Spec Clarity (−0.29). When we control for problem size (comparing only ideas of similar scope), 78–97% of those losses dissolve: Shadow-Frog picks bigger, more ambitious problems, and among ideas of comparable scope it ties Baseline on groundedness.

<figure class="post-figure" markdown="0">
  <div style="background:#f8fafc;border-radius:8px;padding:8px;border:1px solid #e5e7eb;">
    <iframe src="{{ 'figures/shadow-frog/fig6-radar.html' | relative_url }}"
            title="4-axis radar chart comparing Baseline, Shadow-Frog, and Human across Groundedness, Insight, User Impact, and Spec Clarity."
            style="width:100%;max-width:1000px;height:520px;border:0;border-radius:5px;display:block;margin:0 auto;"
            loading="lazy"></iframe>
  </div>
  <figcaption style="text-align:left;font-size:0.92em;color:#555;margin-top:10px;line-height:1.5;">
    <strong>Figure 6 &mdash; Blind-judged quality profile.</strong> Shadow-Frog tilts toward Insight and User Impact; its Groundedness gap largely dissolves when controlling for problem size. Human ideas (amber) are rewritten into the same format agents use before judging, so the comparison reflects substance rather than writing style.
  </figcaption>
</figure>

A follow-up post will explore how to evaluate machine-generated ideas more broadly, stay tuned.

### 5 · Bug fixing

When an agent is *given* a bug to fix (same 50 tasks, re-using the shadows built in §3's dream campaigns, vs Copilot CLI baseline, 3 independent runs), Shadow-Frog resolves **82.0% vs baseline's 77.3%** (+4.7 percentage points), winning **6 of 9 tasks where the two systems disagreed**. However, most of the lift traces to the structured investigate-first workflow the dream skill encourages, not to shadow content itself. Only 1 of the 6 wins shows clear shadow influence (a cross-file document that gave the agent the architectural context to produce a surgical 5-line fix where the baseline rewrote ~80 lines). The bottleneck is not retrieval (§1 already shows ~98% recall) but whether the agent actually reads and acts on the shadow hints it receives: of the 23 tasks where hints were delivered, the agent opened 12, judged 2 relevant, and acted on 1. More aggressive hint inlining is what we are pushing on next.

## What's next

- **Evaluating machine-generated ideas.** Going beyond the metrics from §4 with richer human-authored rubrics, and studying how closely LLM judges align with human assessors when scoring proposed ideas. The goal is a reproducible target for "is this idea actually any good", grounded in human judgment.
- **Long-horizon coding task synthesis.** Compounding dream artifacts (the lineage shown in Figure 0) are a natural source of long, multi-step coding tasks with verifiable execution traces. We are exploring how to turn dream chains into training and evaluation material for the next generation of coding agents.
- **Cross-repo shadows** (branch, repo, org tiers) so an organization-wide shadow can absorb cross-project knowledge, and so a shadow built on one repo can prime work on a related one. We are already dogfooding a version of this on our own MSR Montreal repos: a Shadow-Frog instance watches the team's projects and opens issues on findings it considers high-signal.
- **Shadows beyond source code.** Extend the same discover-by-doing loop to richer, less structured text such as conversation transcripts and design documents. The shadow captures what only emerges through use, not just what is literally written.
- **Cross-language evaluation.** Every benchmark here is Python; we want the same numbers on JavaScript, Rust, Go.
- **Improving hint adherence.** The fixing study shows that even when shadow hints are surfaced right before an edit, the agent acts on only 1 of 23. Better snippet selection, more persuasive presentation, and broader coverage are all on the table.

## Try it

Shadow-Frog is open-source at **[github.com/microsoft/ShadowFrog](https://github.com/microsoft/ShadowFrog)**. Clone the repo, then install into any project with `./install.sh --project /path/to/repo` and invoke `/shadow-frog-init` from your Copilot CLI or Claude Code session. The first dream takes a few minutes; from there, every `/shadow-frog-dream` session you kick off (interactively before stepping away, or wired into your own scheduling / cloud-agent trigger) grows the shadow further.

Shadow-Frog includes six skills and two session hooks.

- **`shadow-frog`**: always-loaded reference docs for the format and conventions
- **`shadow-frog-init`**: onboard a new codebase by scaffolding its empty shadow (run once per repo)
- **`shadow-frog-update`**: refresh the shadow after commits; capture session knowledge
- **`shadow-frog-dream`**: run the discovery loop
- **`shadow-frog-meditate`**: deduplicate, merge, and resolve conflicting discoveries
- **`shadow-frog-viewer`**: browse, search, and audit what's in the shadow

A `sessionStart` hook warns when the shadow is behind the codebase; a `preToolUse` hook inlines actionable-discovery snippets just before every code edit.

If you try it on your own codebase, we'd love to hear what your shadow turns up.

## The landscape

The idea of giving coding agents persistent memory is gaining traction across several fronts. We group related efforts into five threads and then note where Shadow-Frog departs.

**Session-level persistent context.**
Tools like [Cline Memory Bank](https://docs.cline.bot/best-practices/memory-bank), [Windsurf Memories](https://docs.devin.ai/desktop/cascade/memories), [Cursor Rules](https://cursor.com/docs/context/rules), and Claude Code's [CLAUDE.md](https://code.claude.com/docs/en/best-practices) let developers persist project-level instructions across sessions in markdown files. A thriving open-source ecosystem has grown around this pattern: skill frameworks like [Superpowers](https://github.com/obra/superpowers), MCP-based memory servers like [Serena](https://github.com/oraios/serena) and [engram](https://github.com/Gentleman-Programming/engram), and dozens of community Memory Bank ports. These are primarily *user-authored* and *project-scoped*: they tell the agent how to behave, but don't capture what the agent discovers on its own.

**Codebase wikis and structure extraction.**
[DeepWiki](https://docs.devin.ai/work-with-devin/deepwiki) (Cognition, 2025) auto-generates wiki-style documentation from a repo's existing source. [Swimm](https://swimm.io) produces architecture maps and knowledge bases via static analysis. On the academic side, [Aider's repo-map](https://aider.chat/docs/repomap.html) builds a graph-ranked symbol index from tree-sitter parses; [CodexGraph](https://arxiv.org/abs/2408.03910) (Liu et al., 2024) loads code structure into a graph database the agent queries directly; [RepoAgent](https://arxiv.org/abs/2402.16667) (Luo et al., 2024) walks a repo to generate per-function documentation. All of these restructure knowledge that is *already in the source*. As we noted earlier: that is its own large body of work, but as models get stronger it competes against an unusually stiff baseline, namely plain `grep` over the raw code.

**LLM-maintained knowledge wikis.**
Two recent works concurrent with Shadow-Frog articulate the same pattern: instead of re-deriving answers from raw sources on every query (as RAG does), give an LLM a directory of markdown files it owns and incrementally maintains. [Andrej Karpathy's LLM-Wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) makes the case for any large corpus a team repeatedly queries (a personal research notebook, a book companion, an internal team wiki): the human curates the raw sources and the LLM does the bookkeeping (writing summaries, maintaining cross-references, flagging contradictions, keeping an index and a log), so the wiki itself becomes the corpus the model reads from. The [Open Knowledge Format](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing) (Google Cloud) formalizes the same idea as a vendor-neutral specification aimed at enterprise data context (table schemas, business metric definitions, runbooks, deprecation notices): each concept is a single markdown file with a small YAML frontmatter convention (`type`, `title`, `description`, `resource`, `tags`, `timestamp`), and a bundle is just a directory, hostable in any git repo with no SDK or runtime to adopt. Both treat the resulting wiki as a **compounding artifact** rather than a transient retrieval result, and place the maintenance burden on the model rather than the human.

**Memory consolidation and autonomous exploration.**
[Auto-Dreamer](https://arxiv.org/abs/2605.20616) (Ye et al., 2026) introduces a learned offline consolidator (inspired by the brain-science idea that sleep consolidates short-term experience into long-term knowledge): it periodically rewrites a memory bank to deduplicate and abstract across sessions. [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) (Wang et al., 2024) induces reusable workflows from past trajectories. [Claude Dreams](https://platform.claude.com/docs/en/managed-agents/dreams) (Anthropic, 2026) runs asynchronous reflection jobs that reorganize an agent's memory store from session transcripts. These systems consolidate knowledge the agent has *already seen*. [Voyager](https://arxiv.org/abs/2305.16291) (Wang et al., 2023) is different: it explores open-endedly, builds a growing skill library, and discovers things it couldn't have known without acting, but it operates in Minecraft, not in real codebases.

**Parallel agentic orchestration.**
A complementary thread scales a *single* task across many agents at once. [Dynamic workflows](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code) (Anthropic, 2026), introduced alongside Claude Opus 4.8 as we were preparing this post, let Claude Code write orchestration scripts that distribute a user's request across tens to hundreds of parallel subagents in a single session: Claude plans the work, has agents attack from independent angles while adversarial agents try to refute their findings, verifies against the existing test suite, and iterates until the answers converge. Its flagship use cases (i.e., codebase-wide bug hunts, security and performance audits, and large migrations) line up almost exactly with the categories our dream loop explores. In addition, the design rests on the same bet we make: the way to learn useful things about a codebase is to *run* agents against it, rather than to re-read it.

**Where Shadow-Frog departs.**
Most uses of "dream" in this literature mean *offline consolidation*: reorganizing memories the agent already collected while working on tasks. Shadow-Frog's dream is closer to Voyager's spirit: the agent actively explores the codebase by *doing*, implementing features, fuzzing inputs, probing code paths, and the discoveries it distills are knowledge that **did not previously exist anywhere**, not in the source, not in any prior session transcript. The shadow is not a wiki of what the code says; it is a record of what we learned by running experiments against it.

Of the threads above, the LLM-Wiki / OKF pattern is most closely aligned with Shadow-Frog, and we extend it in three codebase-specific ways. **Symbol-level bidirectional anchoring:** every discovery is filed against a `file::symbol` reference, so the wiki mirrors the codebase at the granularity of named entities, and an agent can navigate between code and its corresponding knowledge in a single lookup. **Dual provenance and trust:** each discovery carries a `source:` tag (`exploration`, `user`, `interaction`) and a `verified` / `uncertain` / `refuted` state, so later agents can trade speed for confidence. **Active discovery alongside curation:** general LLM wikis ingest human-curated sources; a shadow also accumulates the byproduct of dreams, knowledge no source document contained.

Dynamic workflows leave out three design choices central to ours. **Persistence.** Their orchestration is an *execution engine* for a task the user hands over: once its agents converge on a single answer, it dissolves and the next session starts cold. A dream instead leaves a durable artifact, the *knowledge* it uncovered, deduplicated, provenance-tagged, and re-read by later agents. **Self-direction.** Rather than executing a task the user specifies, a dream chooses for itself what to investigate during idle time. **Compounding.** Each dream can branch from an earlier one, so the shadow grows across sessions. One orchestrates execution; the other accumulates an open, model-agnostic memory of what running the code taught us.

## Citation

<div class="project-bibtex" markdown="0" style="position: relative; padding-top: 1.5rem;">
  <button type="button" class="button is-primary is-small copy-btn"
    style="position: absolute; top: 1rem; right: 1rem;"
    onclick="navigator.clipboard.writeText(document.getElementById('shadow-frog-bibtex').innerText).then(()=>{this.textContent='Copied ✓';this.classList.add('copied');setTimeout(()=>{this.textContent='Copy';this.classList.remove('copied');},1800);}).catch(()=>{this.textContent='Copy failed';setTimeout(()=>{this.textContent='Copy';},1800);})">
    Copy
  </button>
  <pre class="project-bibtex__code"><code id="shadow-frog-bibtex" style="white-space: pre-wrap;">@misc{shadowfrog,
  title  = {Shadow-Frog: Coding Agents that Dream and Discover},
  url    = {https://microsoft.github.io/debug-gym/blog/2026/06/shadow-frog/},
  author = {Yuan, Xingdi and Vera, Fabio and Moldavskaya, Darya and Singh, Chinmay and Shi, Zhengyan and Caccia, Lucas and Pereira, Matheus and Kim, Minseon and Bowers, Emma and Côté, Marc-Alexandre and Sordoni, Alessandro},
  month  = {June},
  year   = {2026}
}</code></pre>
</div>

## References

1. Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** ICLR 2024. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)

2. OpenAI and SWE-bench team. **Introducing SWE-bench Verified.** August 2024. [openai.com/index/introducing-swe-bench-verified](https://openai.com/index/introducing-swe-bench-verified/)

3. John Yang, Kilian Lieret, Carlos E. Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, Binyuan Hui, Ofir Press, Ludwig Schmidt, and Diyi Yang. **SWE-smith: Scaling Data for Software Engineering Agents.** 2025. [arXiv:2504.21798](https://arxiv.org/abs/2504.21798)

4. GitHub. **Copilot CLI.** 2025. [github.com/features/copilot/cli](https://github.com/features/copilot/cli)

5. Anthropic. **Claude (Opus 4.6, Opus 4.7).** 2025–2026. [anthropic.com](https://www.anthropic.com)

6. OpenAI. **GPT-5.5.** 2026. [openai.com](https://openai.com)

7. Chongrui Ye, Yuxiang Liu, Yu Wang, Haofei Yu, Yining Zhao, Ge Liu, Julian McAuley, and Jiaxuan You. **Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents.** May 2026. [arXiv:2605.20616](https://arxiv.org/abs/2605.20616)

8. Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. **Agent Workflow Memory.** September 2024. [arXiv:2409.07429](https://arxiv.org/abs/2409.07429)

9. Xiangyan Liu, Bo Lan, Zhiyuan Hu, Yang Liu, Zhicheng Zhang, Fei Wang, Michael Shieh, and Wenmeng Zhou. **CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases.** August 2024. [arXiv:2408.03910](https://arxiv.org/abs/2408.03910)

10. Qinyu Luo, Yining Ye, Shihao Liang, Zhong Zhang, Yujia Qin, Yaxi Lu, Yesai Wu, Xin Cong, Yankai Lin, Yingli Zhang, Xiaoyin Che, Zhiyuan Liu, and Maosong Sun. **RepoAgent: An LLM-Powered Open-Source Framework for Repository-level Code Documentation Generation.** February 2024. [arXiv:2402.16667](https://arxiv.org/abs/2402.16667)

11. Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. **Voyager: An Open-Ended Embodied Agent with Large Language Models.** May 2023. [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)

12. Anthropic. **Claude Dreams (Managed Agents).** April 2026. [platform.claude.com/docs/en/managed-agents/dreams](https://platform.claude.com/docs/en/managed-agents/dreams)

13. Cline. **Memory Bank.** 2024. [docs.cline.bot/best-practices/memory-bank](https://docs.cline.bot/best-practices/memory-bank)

14. Windsurf (Cognition). **Cascade Memories.** 2025. [docs.devin.ai/desktop/cascade/memories](https://docs.devin.ai/desktop/cascade/memories)

15. Cursor (Anysphere). **Rules.** 2024. [cursor.com/docs/context/rules](https://cursor.com/docs/context/rules)

16. Anthropic. **Best practices for Claude Code.** 2025. [code.claude.com/docs/en/best-practices](https://code.claude.com/docs/en/best-practices)

17. Jesse Vincent. **Superpowers: A Software Development Methodology for Coding Agents.** 2025. [github.com/obra/superpowers](https://github.com/obra/superpowers)

18. Oraios. **Serena: The IDE for Your Coding Agent (MCP).** 2025. [github.com/oraios/serena](https://github.com/oraios/serena)

19. Gentleman Programming. **engram: Persistent Memory for AI Coding Agents.** 2025. [github.com/Gentleman-Programming/engram](https://github.com/Gentleman-Programming/engram)

20. Cognition. **DeepWiki.** 2025. [docs.devin.ai/work-with-devin/deepwiki](https://docs.devin.ai/work-with-devin/deepwiki)

21. Swimm. **Agentic Code Modernization and Knowledge Base Platform.** 2025. [swimm.io](https://swimm.io)

22. Paul Gauthier. **Aider: Repository Map.** 2023. [aider.chat/docs/repomap.html](https://aider.chat/docs/repomap.html)

23. Anthropic. **Introducing Dynamic Workflows in Claude Code.** May 2026. [claude.com/blog/introducing-dynamic-workflows-in-claude-code](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)

24. Andrej Karpathy. **LLM Wiki.** April 2026. [gist.github.com/karpathy/442a6bf555914893e9891c11519de94f](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)

25. Sam McVeety and Amir Hormati. **Introducing the Open Knowledge Format.** Google Cloud Blog, June 2026. [cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing). Spec repo: [github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf).
