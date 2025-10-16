---
layout: project
title: "debug-gym: A Text-Based Environment for Interactive Debugging"
title_html: '<code>debug-gym</code>: A Text-Based Environment for Interactive Debugging'
description: "A text-based environment for interactive debugging that enables Large Language Models to interactively explore codebases"
authors: "The debug-gym Team"
team_logo: "/static/images/froggy_logo_no_bg_no_border.png"
email: "debug-gym@microsoft.com"
affiliation: "Microsoft Research"
github_url: "https://github.com/microsoft/debug-gym"
arxiv_url: "https://arxiv.org/abs/2503.21557"
bibtex: |
  @article{yuan2025debuggym,
    title={debug-gym: A Text-Based Environment for Interactive Debugging},
    author={Xingdi Yuan, Morgane M Moss, Charbel El Feghali, Chinmay Singh, Darya Moldavskaya, Drew MacPhee, Lucas Caccia, Matheus Pereira, Minseon Kim, Alessandro Sordoni, Marc-Alexandre C\^ot\'e},
    journal={arXiv preprint arXiv:2503.21557},
    year={2025},
    url={https://arxiv.org/abs/2503.21557}
  }
---

## Overview

<div style="text-align: center; margin: 2rem 0;">
  <img src="{{ '/static/images/intro_diagram-2.png' | relative_url }}" alt="debug-gym Overview" style="max-width: 100%; height: auto;">
  <p style="margin-top: 1rem; color: #666; line-height: 1.6;">
    In most existing approaches (shown in <span style="color: black; font-weight: bold;">black</span>), an agent rewrites its code conditioned on error messages obtained from executing its code. <code class="lightly-bold">debug-gym</code> equips the agent with additional tools such as <code class="lightly-bold">pdb</code> (shown in <span style="color: rgb(219, 1, 1); font-weight: bold;">red</span>), so it can interactively seek necessary information from the semantic space hidden behind the code, and therefore have better code repairing performance.
  </p>
</div>

## Abstract

Large Language Models (LLMs) are increasingly relied upon for coding tasks, yet in most scenarios it is assumed that all relevant information can be either accessed in context or matches their training data. We posit that LLMs may benefit from the ability to interactively explore a codebase to gather the information relevant to their task. To achieve this, we present a textual environment, namely `debug-gym`, that can be used to develop LLM agents in an interactive coding setting, bridging the gap between current LLM capabilities and large-scale real-world code generation and debugging requirements. Our environment is lightweight and provides a preset of useful tools, such as a python debugger (`pdb`), designed to facilitate a LLM-based agent's interactive debugging. Beyond coding and debugging tasks, this approach can be generalized to other tasks that would benefit from information-seeking behavior by an LLM agent.

## Video Demo

<div style="text-align: center; margin: 2rem 0;">
  <img src="{{ '/static/images/debug-bench-demo.gif' | relative_url }}" alt="Video Demo">
</div>

## Experiments: Mini Nightmare

Mini-nightmare is a collection of ten minimal buggy Python scripts that human developers would find easier to debug if provided with interactive debugging tools. For instance, the `shopping_cart` problem, which consists of a simple code snippet that aims to implement a shopping cart class. This specific task was created in Python 2, where the round() function adopts an "away from zero" style. However, in Python 3, where we run the debug-gym experiments in, the behavior has been changed to a "ties to even" style. We observe smart debugging traces in trajectories from the agent using Claude 3.7 Sonnet below, showing its use of the pdb tool to test its hypothesis and then import an external library to make sure the rounding is working properly.

<div style="text-align: center; margin: 2rem 0;">
  <img src="{{ '/static/images/shopping_cart_mini_nightmare.png' | relative_url }}" alt="Mini Nightmare Example">
</div>
