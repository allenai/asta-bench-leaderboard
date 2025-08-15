import gradio as gr


def build_page():
    with gr.Column(elem_id="about-page-content-wrapper"):
        # --- Section 1: About AstaBench ---
        gr.HTML(
            """
            <h2>About AstaBench</h2>
            <p>
                AstaBench is a novel AI agents evaluation framework, providing a challenging new test for AI agents: the first benchmark challenge that evaluates agents’ scientific abilities on a broad spectrum of research skills, including literature understanding, data analysis, planning, tool use, coding, and search. Asta’s set of standard tools makes it easy to build general-purpose science agents and to compare their performance in an apples-to-apples manner.
            </p>
            """
        )
        gr.Markdown("---", elem_classes="divider-line")

        # --- Section 2: Why AstaBench? ---
        gr.HTML(
            """
            <h2>Why AstaBench?</h2>
            <p>
                Most current benchmarks test agentic AI and isolated aspects of scientific reasoning, but rarely evaluate AI agentic behavior rigorously or capture the full skill set scientific research requires. Agents can appear effective despite inconsistent results and high compute use, often outperforming others by consuming more resources. Advancing scientific AI requires evaluations that emphasize reproducibility, efficiency, and the real complexity of research.
            </p>
            <br>
            <p>
                AstaBench fills this gap: an agents evaluation framework and suite of open benchmarks for evaluating scientific AI assistants on core scientific tasks that require novel reasoning. AstaBench helps scientists identify which agents best support their needs through task-relevant leaderboards, while giving AI developers a standard execution environment and tools to test the scientific reasoning capabilities of their agents compared to well-known baselines from the literature, including both open and closed LLM foundation models.
            </p>
            """
        )
        gr.Markdown("---", elem_classes="divider-line")

        # --- Section 3: What Does AstaBench Include? ---
        gr.HTML(
            """
            <h2>What Does AstaBench Include?</h2>
            <p>
                AstaBench includes a rigorous agents evaluation framework and a suite of benchmarks consisting of over 2,400 problems across 11 benchmarks, organized into four core categories:
            </p>
            <ul class="info-list">
                <li>Literature Understanding</li>
                <li>Code & Execution</li>
                <li>Data Analysis</li>
                <li>End-to-End Discovery</li>
            </ul>
            <p>
                Plus: a large suite of integrated agents and leaderboards with results from extensive evaluation of agents and models.
            </p>
            <p>
                🔍 Learn more in the AstaBench technical blog post
            </p>
            """
        )
        gr.Markdown("---", elem_classes="divider-line")

        # --- Section 4: Understanding the Leaderboards ---
        gr.HTML(
            """
            <h2>Understanding the Leaderboards</h2>
            <p>
                The AstaBench Overall Leaderboard provides a high-level view of overall agent performance and efficiency:
            </p>
            <ul class="info-list">
                <li>Overall score: A macro-average of the four category-level averages (equal weighting)</li>
                <li>Overall cost: Average cost per task, aggregated only across benchmarks with reported cost</li>
            </ul>
            <p>
                Each category leaderboard provides:
            </p>
            <ul class="info-list">
                <li>Average score and cost for that category (macro-averaged across the benchmarks in the category)</li>
                <li>A breakdown by individual benchmarks</li>
            </ul>
            """
        )
        gr.Markdown("---", elem_classes="divider-line")

        # --- Section 5: Scoring & Aggregation ---
        gr.HTML(
            """
            <h2>Scoring & Aggregation</h2>
            <p>
                AstaBench encourages careful, transparent evaluation. Here's how we handle scoring, cost, and partial results:
            </p>
            
            <h3>Scores</h3>
            <ul class="info-list">
                <li>Each benchmark returns an average score based on per-problem scores</li>
                <li>All scores are aggregated upward using macro-averaging</li>
                <li>Partial completions are included (even with poor performance)</li>
            </ul>
            
            <h3>Cost</h3>
            <ul class="info-list">
                <li>Costs are reported in USD per task.</li>
                <li>Benchmarks without cost data are excluded from cost averages</li>
                <li>In scatter plots, agents without cost are plotted to the far right and clearly marked.</li>
            </ul>
            
            <p>
                <em>Note: Cost values reflect pricing and infrastructure conditions at the time of each submission. We recognize that compute costs may change over time, and are actively working on methods to normalize cost data across submissions for fairer longitudinal comparisons.</em>
            </p>
            
            <h3>Coverage</h3>
            <ul class="info-list">
                <li>Main leaderboard: category coverage (X/4)</li>
                <li>Category view: benchmark coverage (X/Y)</li>
                <li>Incomplete coverage is flagged visually</li>
            </ul>
            
            <p>
                These design choices ensure fair comparison while penalizing cherry-picking and omissions.
            </p>
            """
        )
        gr.Markdown("---", elem_classes="divider-line")

        # --- Section 6: Learn More ---
        gr.HTML(
            """
            <h2>Learn More</h2>
            <ul class="info-list">
                <li>AstaBench technical blog post</li>
                <li>FAQ and submission guide</li>
            </ul>
            """
        )

        # Floating feedback button
        floating_feedback_button_html = """
        <div>
            <a id="feedback-button" href="https://docs.google.com/forms/d/e/1FAIpQLSfJdVkD62aPYh8XehN2FrSeHUWt488Ejc-QdtuZn5NZ3eNoxA/viewform">Have feedback?</a>
        </div>
        """
        gr.HTML(floating_feedback_button_html)