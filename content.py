TITLE = """<h1 align="left" id="space-title">AstaBench Leaderboard</h1>"""

INTRO_PARAGRAPH = """
Newer benchmarks may test agentic AI and isolated aspects of scientific reasoning, but none rigorously measure agentic AI or capture the full range of skills research demands. Agents can appear effective by simply retrying tasks—often at high computational cost and with inconsistent results. Scientific AI needs evaluations that reflect the real complexity of research.
<br>
<br>
AstaBench fills that gap: a suite of open benchmarks for evaluating scientific AI assistants on core scientific tasks that require novel reasoning. The suite includes over 8,000 tasks across 11 benchmarks, organized into four core categories: Literature Understanding, Code & Execution, Data Analysis, and End-to-End Discovery.
<br>
<br>
The **AstaBench Leaderboard** below provides a high-level summary of agent performance and efficiency. It includes:
<br>
<br>
- An **overall score**, computed as a macro average of the four category-level macro averages, ensuring each domain contributes equally—regardless of how many benchmarks each category includes. This provides a fair and balanced comparison across agents with varying capabilities.
- An **overall average cost per task**, consistently aggregated across all categories, to reflect the real efficiency of each agent under comparable conditions.
<br>
To support domain-specific insight, AstaBench also provides per-category leaderboards:
<br>
<br>
- Literature Understanding
<br>
- Code & Execution
<br>
- Data Analysis
<br>
- End-to-End Discovery
<br>
Each category page includes a summary table (average score and cost per problem for that domain), as well as per-benchmark leaderboards for detailed comparisons on specific tasks.
<br>
<br>
🔍 Learn more in the AstaBench technical blog post
"""
SCATTER_DISCLAIMER = """
Note: Only agents with valid cost data are shown in the scatter plot, as both performance and efficiency are required for comparison. Agents without cost data still appear in the tables below.
"""
PARETO_DISCLAIMER = """
Agents names that are green are Pareto optimal, meaning they achieve the best performance for their cost. 
"""
LIT_DESCRIPTION = """
The **Literature Understanding** category evaluates how well agents comprehend and interact with scientific literature—testing their ability to find research papers, assess citation quality, extract information from text, and more.
<br>
The scores shown below reflect performance aggregated across five distinct benchmarks, each targeting a different aspect of literature-based reasoning:
<br>
- PaperFinding Bench – PLACEHOLDER DESCRIPTION
<br>
- ScholarQA Bench2 – PLACEHOLDER DESCRIPTION
<br>
- LitQA2-FT – PLACEHOLDER DESCRIPTION
<br>
- ArxivDIGES Tables-Clean – PLACEHOLDER DESCRIPTION
<br>
<br>
Together, these tasks form a comprehensive evaluation of an agent’s ability to navigate, understand, and reason over scientific publications
"""
CODE_EXECUTION_DESCRIPTION = """
The **Code & Execution** category in AstaBench includes tasks that evaluate an agent’s ability to write, modify, and run code in realistic research scenarios. Unlike literature tasks—which can sometimes be solved by a language model alone—these problems often require the agent to interact with tools: reading input files, executing code, and writing outputs to specific files in the required format.
<br>
<br>
The scores in this category are aggregated from three distinct benchmarks, each targeting different facets of scientific coding and execution:
<br>
- CORE-Bench-Hard – PLACEHOLDER DESCRIPTION
<br>
- DS-1000 – PLACEHOLDER DESCRIPTION
<br>
- SUPER-Expert – PLACEHOLDER DESCRIPTION
<br>
<br>
Together, these benchmarks evaluate whether an agent can function as a hands-on scientific assistant—not just by reasoning about code, but by running it in real-world contexts.
"""
DATA_ANALYSIS_DESCRIPTION = """
The **Data Analysis** category evaluates agents on their ability to analyze structured datasets and generate meaningful scientific hypotheses. It currently includes a single benchmark:
<br>
 - DiscoveryBench
<br>
so the category-level scores are the same as the benchmark-level results.
<br>
<br>
As additional benchmarks are added in the future, this category will expand to cover a broader range of data-driven reasoning tasks across scientific domains.
"""
DISCOVERY_DESCRIPTION = """
The **End-to-End Discovery** category tests whether agents can carry out a complete scientific workflow—from hypothesis generation and experiment design to code execution, analysis, and report writing. These tasks require agents to integrate multiple capabilities, producing not just answers but full research artifacts.
<br>
<br>
Scores in this category are aggregated from two benchmarks:
<br>
- E2E-Bench – PLACEHOLDER DESCRIPTION
<br>
- E2E-Bench-Hard – PLACEHOLDER DESCRIPTION
<br>
<br>
This category provides the first standardized way to evaluate automated scientific discovery (ASD) agents across all stages of the research process.
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""@article{asta-bench,
    title={AstaBench},
    author={AstaBench folks},
    year={2025},
    eprint={TBD.TBD},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    secondaryClass={cs.CL}
}"""
# legend_tooltips = {
#     "pareto": "The Pareto frontier represents optimal agents where you cannot improve score without increasing cost.",
#     "openness": "Describes the accessibility of the agent's core model (e.g., Open, Closed, API).",
#     "tooling": "Describes the tools an agent uses (e.g., Standard, Custom)."
# }

def format_error(msg):
    return f"<p style='color: red; font-size: 20px; text-align: center;'>{msg}</p>"


def format_warning(msg):
    return f"<p style='color: orange; font-size: 20px; text-align: center;'>{msg}</p>"


def format_log(msg):
    return f"<p style='color: green; font-size: 20px; text-align: center;'>{msg}</p>"


def hyperlink(link_url: str, text: str = "🔗") -> str:
    if not link_url or not isinstance(link_url, str):
        return str(text) # Or simply "" if link_url is bad
    # Using a simpler style here for broad compatibility, your original style is fine too.
    return f'<a target="_blank" href="{link_url}">{text}</a>'


def hf_uri_to_web_url(uri: str) -> str:
    """
    Convert a Hugging Face-style URI like:
        hf://datasets/{namespace}/{repo}/{path...}
    into a public web URL:
        https://huggingface.co/datasets/{namespace}/{repo}/tree/main/{path...}
    """
    prefix = "hf://datasets/"
    if not uri.startswith(prefix):
        raise ValueError("URI must start with 'hf://datasets/'")

    parts = uri[len(prefix) :].split("/", 2)
    if len(parts) < 3:
        raise ValueError("Expected format: hf://datasets/{namespace}/{repo}/{path...}")

    namespace, repo, path = parts
    return f"https://huggingface.co/datasets/{namespace}/{repo}/tree/main/{path}"

css = """
#intro-paragraph {
    font-size: 18px;
    max-width: 60%;
}
#category-intro {
    font-size: 18px;
    max-width: 60%;
}
#logo-image { 
    margin: 0;
    margin-bottom: 30px; 
    justify-content: flex-start;        
    max-width: 250px;       
    height: auto;           
}
.table-component{
    height: auto !important;
    max-height: none !important;
}
.table-wrap {
    max-height: none !important;
    height: auto !important;
    overflow-y: visible !important;
}
/* --- New Rules for Table Density --- */
table.gr-table th, table.gr-table td {
    padding: 4px 4px !important; 
    width: 1%;
    white-space: nowrap;
}
table.svelte-1e98i6s td {
    vertical-align: top !important;
}
table.gr-table {
    font-size: 14px !important;
}
.html-container {
    padding-top: 0 !important;
}
#scatter-disclaimer {
    color: #f0529c !important;
}
#pareto-disclaimer {
    color: #f0529c !important;
}
thead.svelte-1e98i6s th {
    background: white !important;
}
.dark thead.svelte-1e98i6s th {
    background: #091a1a !important;
}
.cell-wrap.svelte-v1pjjd {
    font-family: 'Manrope';
    }
nav.svelte-ti537g.svelte-ti537g {
    justify-content: flex-start;
}
#legend-markdown span {
    margin-right: 15px !important; 
}
#leaderboard-accordion .label-wrap {
    font-size: 1.4rem !important; 
    z-index: 10 !important;
    position: relative !important;
}
.dark #leaderboard-accordion .label-wrap {
    color: #0FCB8C !important; 
}
.dark block.svelte-1svsvh2 {
    background: #032629 !important;
}
.padding.svelte-phx28p {
    padding: 0 !important;
}
.dark .sub-nav-link-button {
    color: #0fcb8c !important;
}
.sub-nav-bar-container {
    display: flex !important;
    flex-wrap: nowrap !important; 
    align-items: center !important; 
    gap: 20px !important;
}
.sub-nav-link-button {
    background: none;
    border: none;
    padding: 0;
    margin: 0;
    font-family: inherit;
    font-size: 16px;
    color: #F263A6; 
    text-decoration: none;
    cursor: pointer;
    white-space: nowrap;
}
.sub-nav-link-button:hover {
    text-decoration: underline;
}
.wrap-header-df th span{
    white-space: normal !important;
    word-break: normal !important;
    overflow-wrap: break-word !important;
    line-height: 1.2 !important;
    vertical-align: top !important;
    font-size: 12px !important;
    
}
.wrap-header-df th {
    height: auto !important;
}
.wrap-header-df .cell-wrap img {
    width: 16px;
    height: 16px;
    vertical-align: middle;
}

/* By default, hide BOTH theme-aware icons inside a DataFrame cell */
.wrap-header-df .cell-wrap .light-mode-icon,
.wrap-header-df .cell-wrap .dark-mode-icon {
    display: none !important;
}

/* Light Theme Rule: Show the light-mode icon */
html:not(.dark) .wrap-header-df .cell-wrap .light-mode-icon {
    display: inline-block !important;
}

/* Dark Theme Rule: Show the dark-mode icon */
.dark .wrap-header-df .cell-wrap .dark-mode-icon {
    display: inline-block !important;
}
#legend-markdown img {
    width: 16px;
    height: 16px;
    vertical-align: middle;
}
html:not(.dark) #legend-markdown .light-mode-icon,
.dark #legend-markdown .dark-mode-icon {
    display: inline-block;
}
#legend-markdown .light-mode-icon, #legend-markdown .dark-mode-icon {
    display: none;
}

/* Column description tooltip styles */
#legend-markdown,
#leaderboard-accordion {
    overflow: visible !important;
}

.tooltip-icon {
    display: inline-block;
    margin-left: 6px;
    cursor: help;
    position: relative;
}

.tooltip-icon::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 125%;
    background-color: #333;
    color: #fff;
    padding: 12px 16px;
    border-radius: 4px;
    font-size: 12px;
    opacity: 0;
    transition: opacity 0.2s;
    white-space: pre-line;
    width: 500px;
    text-align: left;
    pointer-events: none;
}

.tooltip-icon:hover::after {
    opacity: 1;
}
"""
