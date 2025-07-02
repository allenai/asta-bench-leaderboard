TITLE = """<h1 align="left" id="space-title">AstaBench Leaderboard</h1>"""

INTRO_PARAGRAPH = """
AI agents are on the rise, promising everything from travel planning to scientific discovery. But evaluating them—especially for real-world research tasks—remains a messy, inconsistent process. Metrics vary, cost is often ignored, and scientific use cases are rarely the focus. <br>
<br>
Enter AstaBench, a grand challenge benchmark developed by Ai2 to test how well agentic AI systems perform on scientific tasks that actually matter. As part of the Asta initiative, AstaBench spans ten multi-step benchmarks covering literature review, data analysis, code execution, and complex decision-making. It brings standardization and transparency to agent evaluation, with statistical confidence reporting, and a leaderboard that highlights tradeoffs between accuracy and computational cost.
"""
SCATTER_DISCLAIMER = """
Only agents that have cost data available will be shown in the scatter plot. If you don't see your agent, please ensure that you have provided cost data in your submission.
"""
PARETO_DISCLAIMER = """
Agents names that are green are Pareto optimal, meaning they achieve the best performance for their cost. 
"""

SUBMISSION_TEXT = """
## Submissions
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
.submission-accordion {
    border-style: solid;
    border-width: 3px !important;
    border-color: #ec4899;
}
.submission-accordion span.svelte-1w6vloh { 
    font-weight: bold !important;
    font-size: 1.2em !important;
}
#logo-image { 
    margin: auto;          
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

table.gr-table {
    font-size: 14px !important;
}

/* Example of making the "Agent" column (the 1st column) a bit wider if needed */
table.gr-table th:nth-child(1),
table.gr-table td:nth-child(1) {
    min-width: 150px !important;
    white-space: normal !important; /* Allow agent names to wrap if long */
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
"""
