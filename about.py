import gradio as gr


def build_page():
    gr.Markdown(
        """
## About AstaBench
AstaBench is a novel AI agents evaluation framework, providing a challenging new test for AI agents: the first benchmark challenge that evaluates agents’ scientific abilities on a broad spectrum of research skills, including literature understanding, data analysis, planning, tool use, coding, and search. Asta’s set of standard tools makes it easy to build general-purpose science agents and to compare their performance in an apples-to-apples manner.


## Why AstaBench?
Most current benchmarks test agentic AI and isolated aspects of scientific reasoning, but rarely evaluate AI agentic behavior rigorously or capture the full skill set scientific research requires. Agents can appear effective despite inconsistent results and high compute use, often outperforming others by consuming more resources. Advancing scientific AI requires evaluations that emphasize reproducibility, efficiency, and the real complexity of research.

AstaBench fills this gap: an agents evaluation framework and suite of open benchmarks for evaluating scientific AI assistants on core scientific tasks that require novel reasoning. AstaBench helps scientists identify which agents best support their needs through task-relevant leaderboards, while giving AI developers a standard execution environment and tools to test the scientific reasoning capabilities of their agents compared to well-known baselines from the literature, including both open and closed LLM foundation models.


## What Does AstaBench Include?
AstaBench includes a rigorous agents evaluation framework and a suite of benchmarks consisting of over 2,400 problems across 11 benchmarks, organized into four core categories:
Literature Understanding
Code & Execution
Data Analysis
End-to-End Discovery
Plus: a large suite of integrated agents and leaderboards with results from extensive evaluation of agents and models.

🔍 Learn more in the AstaBench technical blog post


## Understanding the Leaderboards
The AstaBench Overall Leaderboard provides a high-level view of overall agent performance and efficiency:
- Overall score: A macro-average of the four category-level averages (equal weighting)
- Overall cost: Average cost per task, aggregated only across benchmarks with reported cost

Each category leaderboard provides:
- Average score and cost for that category (macro-averaged across the benchmarks in the category)
- A breakdown by individual benchmarks


## Scoring & Aggregation
AstaBench encourages careful, transparent evaluation. Here's how we handle scoring, cost, and partial results:

**Scores**
- Each benchmark returns an average score based on per-problem scores
- All scores are aggregated upward using macro-averaging
- Partial completions are included (even with poor performance)

**Cost**
- Costs are reported in USD per task.
- Benchmarks without cost data are excluded from cost averages
- In scatter plots, agents without cost are plotted to the far right and clearly marked. 

Note: Cost values reflect pricing and infrastructure conditions at the time of each submission. We recognize that compute costs may change over time, and are actively working on methods to normalize cost data across submissions for fairer longitudinal comparisons.

**Coverage**
- Main leaderboard: category coverage (X/4)
- Category view: benchmark coverage (X/Y)
- Incomplete coverage is flagged visually

These design choices ensure fair comparison while penalizing cherry-picking and omissions.


## Learn More
- AstaBench technical blog post
- FAQ and submission guide
""", elem_id="about-content"
    )