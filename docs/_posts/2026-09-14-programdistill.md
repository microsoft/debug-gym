---
layout: blog-post
title: "ProgramDistill: From Interactive Web Apps to Verifiable Reference-Guided SWE Tasks"
date: 2026-09-14
published: false
author: "Jeonghye Kim"
reading_time: 9
tags: ["Coding Agents", "Reference-Guided SWE", "Web Applications", "Synthetic Tasks", "Benchmark"]
description: "A benchmark and synthetic task-generation framework that turns working web applications into replay-verifiable SWE tasks and evaluates whether coding agents can recover missing behavior from a live reference."
paper_local: "/static/papers/ProgramDistill_arxiv.pdf"
version_local_assets: true
explorer_manifest: "/figures/programdistill/explorer/cases.json"
authors:
  - name: "Jeonghye Kim"
    role: "KAIST, Research Intern at MSR Montr\u00e9al"
---

<link rel="stylesheet" href="{{ '/static/css/programdistill-post.css' | relative_url }}">

## TL;DR

{% assign demo_file = site.static_files | where: "path", "/static/images/programdistill-demo.mp4" | first %}
{% assign demo_version = demo_file.modified_time | date: "%s" %}
<video class="pd-demo" width="1920" height="1080" controls muted playsinline preload="metadata" poster="{{ '/static/images/programdistill-demo-poster.jpg' | relative_url }}?v=20260914-title" aria-label="ProgramDistill demo" markdown="0">
  <source src="{{ '/static/images/programdistill-demo.mp4' | relative_url }}?v={{ demo_version }}" type="video/mp4">
  <a href="{{ '/static/images/programdistill-demo.mp4' | relative_url }}?v={{ demo_version }}" download>Download the ProgramDistill demo</a>
</video>

<blockquote class="pd-summary-box" markdown="0">
  <p><strong>A benchmark for reference-guided web development.</strong> Working software serves as both the <strong>specification and verifier</strong>. Coding agents inspect a reference app, infer its behavior, restore it in an editable app, and validate the result.</p>
  <p><span aria-hidden="true">🧩</span> <strong>Apps become verifiable SWE tasks.</strong> ProgramDistill turns replayable behaviors into repair tasks while preserving their prerequisite structure.</p>
  <p><span aria-hidden="true">📈</span> <strong>Difficulty scales with dependencies.</strong> Restoration depth controls how many behaviors must be restored together.</p>
  <p>Across 26 applications, the pipeline discovers <strong>1,975 replay-verified behaviors</strong> and constructs <strong>4,063 tasks</strong>.</p>
</blockquote>

## From a working reference to working code

In practice, web development does not always begin with a complete specification. The intended behavior may instead be demonstrated by an earlier product version, an interactive prototype, another application, or a working reference.

A developer can interact with that reference to fill in the missing details. They can observe what changes after an action, which behaviors depend on earlier state, what persists, and how a workflow unfolds. The goal is then to reproduce that behavior in the application being developed.

<figure class="post-figure" markdown="0">
  <a href="{{ '/figures/programdistill/concept.png' | relative_url }}">
    <img src="{{ '/figures/programdistill/concept.png' | relative_url }}" alt="A coding agent interacts with a working reference whose source code is hidden, implements the required behavior in an editable current app, and validates the result." loading="lazy">
  </a>
  <figcaption><strong>Figure 1.</strong> Reference-guided software engineering with a working reference and an editable current application.</figcaption>
</figure>

ProgramDistill turns this workflow into a benchmark. The agent can interact with the working reference but cannot inspect its source code. It edits a current application with missing functionality and is evaluated on whether the resulting application reproduces the reference behavior when the same workflow is executed. The implementation itself does not need to match the original source code.

## A synthetic task generation method built for stateful web applications

To evaluate this workflow at scale, each task needs both a reproducible behavior and a reliable verifier. Constructing such tasks requires exploring a working application, establishing prerequisite state, identifying successful outcomes, and removing the corresponding implementation without breaking earlier behaviors.

ProgramDistill automates this process with the **mine-craft-patch** pipeline, orchestrated by multiple LLM agents. Starting from a working application and its source code, the pipeline discovers behaviors, records them as replayable traces, and removes the source implementations responsible for those behaviors to create repair tasks.

<p class="pd-trace-definition" markdown="0">Each <button class="pd-trace-link" type="button" data-pd-trace-examples="programdistill-explorer" data-traces-src="{{ '/figures/programdistill/explorer/trace-examples.json' | relative_url }}" aria-haspopup="dialog">trace</button> stores browser actions, expected success signals, and an optional prerequisite link. Because the actions can be replayed and the expected signals checked automatically, the trace itself serves as an executable behavioral verifier.</p>

<ol class="pd-process" aria-label="Task generation and evaluation pipeline" role="list" markdown="0">
  <li>
    <span class="pd-process-number" aria-hidden="true">01</span>
    <h3 id="mine">Mine</h3>
    <p>Mining agents explore the application and record replayable traces. Verified traces can seed the discovery of dependent behaviors, and only reproducible traces are retained.</p>
  </li>
  <li>
    <span class="pd-process-number" aria-hidden="true">02</span>
    <h3 id="craft">Craft</h3>
    <p>Crafting agents remove the implementation of selected behaviors. Build and replay checks ensure that prerequisites still work, the target behavior fails, and the gold patch restores it.</p>
  </li>
  <li>
    <span class="pd-process-number" aria-hidden="true">03</span>
    <h3 id="patch">Patch</h3>
    <p>Coding agents repair the current application by interacting with the working reference, without access to its source code or the gold patch. Replay then evaluates the repair.</p>
  </li>
</ol>

**The same replay mechanism checks the original app, the masked app, and the repair.** It executes the recorded actions and checks the expected signals without LLM intervention.

### Walk through the pipeline!

Choose one of the three examples below and follow the **mine-craft-patch** pipeline from mined behaviors to masks and saved repair runs. The explorer shows recorded examples rather than launching new agent runs. Each application is repository-based, with its GitHub source linked below the example list.

<p class="pd-full-dashboard-cta"><button class="pd-full-dashboard-link" data-pd-dashboard-soon="programdistill-explorer" type="button" aria-haspopup="dialog">Explore all applications in the full dashboard (Coming Soon)</button></p>

{% include programdistill-explorer.html %}

## Product workflows become the curriculum

Product features rarely stand alone. A card must exist before it can be moved or edited, and creating that card may first require a board and a list. Reproducing a later behavior therefore means recreating the state established by earlier ones.

Mining preserves these relationships as **prerequisite lineages**. Each verified trace can build on the state produced by an earlier trace, forming branching dependency trees across applications.

<figure class="post-figure pd-lineage-figure" markdown="0">
  <div class="pd-lineage-panels">
    <a href="{{ '/figures/programdistill/lineage-forest.png' | relative_url }}" title="View the full-size lineage forest">
      <img src="{{ '/figures/programdistill/lineage-forest.png' | relative_url }}" width="1424" height="1492" alt="Mined prerequisite lineage forests across all 26 benchmark applications." loading="lazy">
    </a>
    <a href="{{ '/figures/programdistill/lineage-streamview.png' | relative_url }}" title="View the full-size StreamView lineage">
      <img src="{{ '/figures/programdistill/lineage-streamview.png' | relative_url }}" width="2638" height="1260" alt="A StreamView prerequisite lineage with columns for depths 1 through 12 and one root-to-leaf chain highlighted." loading="lazy">
    </a>
  </div>
  <figcaption><strong>Figure 2.</strong> Prerequisite lineages across 26 applications and a StreamView lineage reaching depth 12.</figcaption>
</figure>

Crafting turns this structure into two kinds of task. An **atomic task** removes one behavior while leaving its prerequisites intact. A **cumulative task** combines accepted masks along one lineage, requiring the agent to restore several connected behaviors together.

<div class="table-container" role="region" aria-label="Generated repair task types" tabindex="0" markdown="0">
  <table class="table is-fullwidth">
    <caption class="is-sr-only">The generated corpus contains 2,862 atomic tasks and 1,201 cumulative tasks.</caption>
    <thead>
      <tr><th scope="col">Task type</th><th scope="col">What the agent restores</th><th scope="col" class="has-text-right">Tasks</th></tr>
    </thead>
    <tbody>
      <tr><th scope="row">Atomic</th><td>One missing behavior</td><td class="has-text-right">2,862</td></tr>
      <tr><th scope="row">Cumulative</th><td>Several behaviors along one prerequisite lineage</td><td class="has-text-right">1,201</td></tr>
    </tbody>
  </table>
</div>

## Partial-Application Reconstruction

Nine coding agents tackle **ProgramDistill-300**, a suite of 300 tasks across 26 applications at restoration depths 1 through 8. Each starts from a mostly working app with missing functionality. Under binary scoring, a cumulative task succeeds only if every repair target in the workflow passes.

**Astra solves every depth-1 task, establishing a strong baseline for individual repair solvability. Yet success drops to 64.0% at depth 8, when multiple dependent behaviors must be restored together.** Every other agent retains less than half of its depth-1 performance at depth 8.

<figure class="post-figure" markdown="0">
  <a href="{{ '/figures/programdistill/repair-results-binary.png' | relative_url }}?v=20260915-trajectory-cost">
    <img src="{{ '/figures/programdistill/repair-results-binary.png' | relative_url }}?v=20260915-trajectory-cost" width="2981" height="840" alt="Partial reconstruction across nine coding agents, plotted against mean cost per trajectory in USD and restoration depth. Overall success is 84.3% for Astra, 68.7% for Opus 5, and 60.7% for Sol. From depth 1 to 8, Astra falls from 100% to 64.0%, Opus from 96.0% to 32.0%, and Sol from 92.0% to 32.0%." loading="lazy">
  </a>
  <figcaption><strong>Figure 3.</strong> Mean binary score versus mean cost per trajectory in USD (left) and by restoration depth (right).</figcaption>
</figure>

### The strongest agent observes more and edits less

Astra is distinctive for an **observation-intensive, edit-light workflow**. It performs substantially more reference and current-app observations than the other models, with the difference particularly pronounced for current-app observation, while making the fewest edit/write steps. It averages **96.3 current-app observations** per trajectory, roughly twice Sol's 45.8, alongside **only 9.9 edit/write steps**. Its trajectories are thus characterized by extensive behavioral checking, especially of its own implementation, followed by comparatively selective code changes. More broadly, the strongest-performing model differs not only in repair accuracy but also in how it allocates effort across observation, validation, and editing.

<figure class="post-figure" markdown="0">
  <a href="{{ '/figures/programdistill/agent-behavior-binary.png' | relative_url }}">
    <img src="{{ '/figures/programdistill/agent-behavior-binary.png' | relative_url }}" width="4745" height="916" alt="Across nine coding agents, Astra has the highest repair accuracy, the most reference and current-app observation steps, and the fewest edit/write steps." loading="lazy">
  </a>
  <figcaption><strong>Figure 4.</strong> Repair accuracy and observation, reading, and editing activity across agents.</figcaption>
</figure>

### Deeper tasks receive less checking per behavior

As restoration depth increases, reconstruction burden grows roughly linearly as agents must recover more interdependent behaviors along a prerequisite lineage. At depth 8, for example, **eight behaviors must be restored in sequence**, with later behaviors relying on state established by earlier ones.

From depth 1 to depth 8, the amount of code to restore grows by **more than 9&times;**, while the total number of actions across target behavior traces grows by **more than 10&times;**. Agent effort, however, does not keep pace. Although agents do more work overall, **observation and editing effort per repair target decline as tasks deepen**, with observation shrinking particularly sharply. Final patches also leave more of the masked implementation unrestored.

<figure class="post-figure" markdown="0">
  <a href="{{ '/figures/programdistill/depth-burden-vs-effort.png' | relative_url }}?v=20260915-behavior-traces">
    <img src="{{ '/figures/programdistill/depth-burden-vs-effort.png' | relative_url }}?v=20260915-behavior-traces" width="3705" height="859" alt="As restoration depth increases, code and action counts across target behavior traces grow, observation and editing steps per behavior fall, and more target files and mask stubs remain unrestored." loading="lazy">
  </a>
  <figcaption><strong>Figure 5.</strong> Growing restoration burden, declining effort per behavior, and more implementation left unrestored.</figcaption>
</figure>

### Example: How an agent completes a depth-8 repair

In [vdevired/trello-clone](https://github.com/vdevired/trello-clone), Claude Opus 5 successfully completes a depth-8 task spanning eight dependent workflow stages, from login and project setup through board and card operations, movement, and comment editing.

The repair takes **317 agent steps**. Rather than solving the task in a single pass, the agent repeatedly inspects the reference, edits the current implementation, checks the repaired application, and returns to remaining mismatches. The trajectory looks much more like iterative development against a live reference than one-shot code generation.

<figure class="post-figure" markdown="0">
  <a href="{{ '/figures/programdistill/trello-repair.png' | relative_url }}">
    <img src="{{ '/figures/programdistill/trello-repair.png' | relative_url }}" alt="A successful 317-step repair of the Trello clone alternates between observing the reference, editing code, and validating the current app." loading="lazy">
  </a>
  <figcaption><strong>Figure 6.</strong> Claude Opus 5 completes a depth-8 repair through repeated reference observation, implementation, and validation.</figcaption>
</figure>

## Full-Application Reconstruction

Full reconstruction removes the existing implementation as a starting point. The agent receives a minimal executable scaffold, a product-level capability description, and browser access to the working reference. It must rebuild the application, which is then evaluated using the replay-verified behaviors discovered during mining.

Across 12 applications, each agent is evaluated on **590 individual behaviors** and **413 cumulative workflows**. Individual tests measure behavior recovery, while a cumulative workflow passes only when all of its targets pass.

<div class="table-container" role="region" aria-label="Full reconstruction results" tabindex="0" markdown="0">
  <table class="table is-fullwidth">
    <caption class="is-sr-only">Full-application reconstruction recovery rates across 12 applications.</caption>
    <thead>
      <tr><th scope="col">Agent</th><th scope="col" class="has-text-right">Individual behaviors</th><th scope="col" class="has-text-right">Cumulative workflows</th></tr>
    </thead>
    <tbody>
      <tr><th scope="row">GPT-6 Astra (max)</th><td class="has-text-right"><strong>58.98%</strong></td><td class="has-text-right"><strong>49.15%</strong></td></tr>
      <tr><th scope="row">Claude Opus 5 (max)</th><td class="has-text-right">42.03%</td><td class="has-text-right">28.81%</td></tr>
      <tr><th scope="row">GPT-5.6 Sol (max)</th><td class="has-text-right">33.39%</td><td class="has-text-right">21.07%</td></tr>
    </tbody>
  </table>
</div>

Even the strongest agent recovers only **49.2%** of cumulative workflows. All three agents recover a larger share of individual behaviors than cumulative workflows, showing that implementing useful pieces does not guarantee that those pieces work together. These runs average roughly 700 agent steps and reach as high as **1,921 steps**.

### Failure analysis: What agents miss

Reference exploration remains a major bottleneck. The failure analysis covers **977 failed atomic behaviors across 36 reconstruction runs**. All percentages below are calculated over these 977 failures.

<dl class="pd-findings" markdown="0">
  <div>
    <dt>Reference coverage</dt>
    <dd><strong>59.2%</strong> involve behaviors that were never observed in the reference.</dd>
  </div>
  <div>
    <dt>Faithful implementation</dt>
    <dd><strong>27.9%</strong> produce the wrong state, route, or result, <strong>11.1%</strong> have the wrong observable form, and <strong>1.8%</strong> were observed but not implemented.</dd>
  </div>
</dl>

These failures point to gaps in both reference exploration and faithful implementation. The largest category consists of behaviors the agent never observed, but the remaining failures show that observing a behavior is not enough to reproduce it correctly.

**An application that runs is not necessarily an application that matches the reference.** Syntax checks, successful backend requests, or selected text matches can miss an incorrect state transition or a workflow that was never rechecked after the final relevant edit. This echoes partial repair, where current-app observation per target falls as tasks deepen.

### Example: A working feature can still be wrong

In one reconstruction of [knowankit/trello-clone](https://github.com/knowankit/trello-clone), Astra implements card dragging, and the card visibly moves. However, after the same center drop, the resulting card order differs from the working reference.

The interaction works, but the resulting state is wrong. The error remains because after its final relevant edit, the agent validates other board interactions instead of replaying the exact drag workflow that would expose the mismatch.

<figure class="post-figure" markdown="0">
  <a href="{{ '/figures/programdistill/astra-trello-order.png' | relative_url }}">
    <img src="{{ '/figures/programdistill/astra-trello-order.png' | relative_url }}" alt="Full reconstruction failure in the Trello clone where the same card drop produces a different ordering from the reference." loading="lazy">
  </a>
  <figcaption><strong>Figure 7.</strong> The reference and Astra's reconstructed application produce different card orders after the same drag interaction.</figcaption>
</figure>

Full reconstruction also shows that more browser feedback alone is not enough. Claude Opus 5 receives substantially more browser-state feedback than Astra, yet recovers fewer behaviors. What matters is not only how much of the reference the agent inspects, but whether it identifies the right details, implements them faithfully, and rechecks the relevant workflow after making changes.

## What's next?

<blockquote class="pd-next-box" markdown="0">
  <p><span aria-hidden="true">🌱</span> <strong>Train reference-guided agents with a curriculum.</strong> Use restoration depth to organize replay-verifiable tasks from simpler repairs to deeper workflows, with agent trajectories for distillation and replay-based rewards for reinforcement learning.</p>
  <p><span aria-hidden="true">👁️</span> <strong>Extend ProgramDistill to multimodal agents.</strong> Add screenshot-based reference interaction and evaluate visual fidelity alongside behavioral correctness.</p>
</blockquote>