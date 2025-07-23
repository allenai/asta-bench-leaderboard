import gradio as gr


with gr.Blocks() as demo:
    gr.Markdown(
        """
## About AstaBench
AstaBench is a best-in-class AI agents evaluation framework to measure scientific research abilities. AstaBench provides a challenging new test for AI agents: the first benchmark challenge that evaluates agents’ scientific abilities on a broad spectrum of research skills, including literature understanding, data analysis, planning, tool use, coding, and search.


**Why AstaBench?**
Newer benchmarks may test agentic AI and isolated aspects of scientific reasoning, but none rigorously measure agentic AI or capture the full range of skills research demands. Agents can appear effective by simply retrying tasks—often at high computational cost and with inconsistent results. Scientific AI needs evaluations that reflect the real complexity of research.

AstaBench fills that gap: a suite of open benchmarks for evaluating scientific AI assistants on core scientific tasks that require novel reasoning. AstaBench helps scientists identify which agents best support their needs through task-relevant leaderboards, while giving AI developers a standard execution environment and standard tools to test the scientific reasoning capabilities of their agents compared to well-known baselines from the literature, including both open and closed LLM foundation models.


**What Does AstaBench Include?**
The suite includes over 8,000 tasks across 11 benchmarks, organized into four core categories:
- Literature Understanding
- Code & Execution
- Data Analysis
- End-to-End Discovery

🔍 Learn more in the AstaBench technical blog post


**Understanding the Leaderboards**
The AstaBench Main Leaderboard provides a high-level view of overall agent performance and efficiency:
- Overall score: A macro-average of the four category-level averages (equal weighting)
- Overall cost: Average cost per task, aggregated only across benchmarks with reported cost

Each category leaderboard provides:
- Average score and cost for that category
- A breakdown by individual benchmarks


**Scoring & Aggregation**
AstaBench encourages broad, honest evaluation. Here's how we handle scoring, cost, and partial results:

_Scores_
- Each benchmark returns an average score based on per-task accuracy
- Skipped benchmarks receive a score of 0.00
- All scores are aggregated upward using macro-averaging
- Partial completions are included (even with poor performance)

_Cost_
- Costs are reported in USD per task, based on values at the time of submission
- Benchmarks without cost data are excluded from cost averages
- In scatter plots, agents without cost are plotted far right and clearly marked
Note: Cost values reflect pricing and infrastructure conditions at the time of each submission. We recognize that compute costs may change over time, and are actively working on methods to normalize cost data across submissions for fairer longitudinal comparisons.


_Coverage_
- Main leaderboard: category coverage (X/4)
- Category view: benchmark coverage (X/Y)
- Incomplete coverage is flagged visually

These design choices ensure fair comparison while penalizing cherry-picking and omissions.


**Learn More**
- AstaBench technical blog post
- FAQ and submission guide
""", elem_id="about-content"
    )