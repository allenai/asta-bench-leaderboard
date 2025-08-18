TITLE = """<h1 align="left" id="space-title">AstaBench Leaderboard</h1>"""

INTRO_PARAGRAPH = """
<p>
    Newer benchmarks may test agentic AI and isolated aspects of scientific reasoning, but none rigorously measure agentic AI or capture the full range of skills research demands. Agents can appear effective by simply retrying tasks—often at high computational cost and with inconsistent results. Scientific AI needs evaluations that reflect the real complexity of research.
</p>
<br>
<p>
    AstaBench fills that gap: a suite of open benchmarks for evaluating scientific AI assistants on core scientific tasks that require novel reasoning. The suite includes over 8,000 tasks across 11 benchmarks, organized into four core categories: Literature Understanding, Code & Execution, Data Analysis, and End-to-End Discovery.
</p>
<br>
<p>
    The <strong>AstaBench Leaderboard</strong> below provides a high-level summary of agent performance and efficiency. It includes:
</p>
<ul class="info-list">
    <li>
        An <strong>overall score</strong>, computed as a macro average of the four category-level macro averages, ensuring each domain contributes equally—regardless of how many benchmarks each category includes. This provides a fair and balanced comparison across agents with varying capabilities.
    </li>
    <li>
        An <strong>overall average cost per task</strong>, consistently aggregated across all categories, to reflect the real efficiency of each agent under comparable conditions.
    </li>
</ul>
<br>
<p>
    To support domain-specific insight, AstaBench also provides per-category leaderboards:
</p>
<ul class="info-list">
    <li>Literature Understanding</li>
    <li>Code & Execution</li>
    <li>Data Analysis</li>
    <li>End-to-End Discovery</li>
</ul>
<br>
<p>
    Each category page includes a summary table (average score and cost per problem for that domain), as well as per-benchmark leaderboards for detailed comparisons on specific tasks.
</p>
<p>
    🔍 Learn more in the AstaBench technical blog post
</p>
"""
SCATTER_DISCLAIMER = """
**Note:** Agents without cost data are displayed to the right of the vertical divider line.
"""
PARETO_DISCLAIMER = """
Agents names that are green are Pareto optimal, meaning they achieve the best performance for their cost. 
"""
LIT_DESCRIPTION = """
The **Literature Understanding** category evaluates how well agents comprehend and interact with scientific literature—testing their ability to find research papers, assess citation quality, extract information from text, and more.
<br><br>
The scores shown below reflect performance aggregated across five distinct benchmarks, each targeting a different aspect of literature-based reasoning. 
<br><br>
For detailed results, use the links above to explore individual benchmarks.
<br>
"""
CODE_EXECUTION_DESCRIPTION = """
The **Code & Execution** category in AstaBench includes tasks that evaluate an agent’s ability to write, modify, and run code in realistic research scenarios. Unlike literature tasks—which only require read-only tools and can sometimes even be solved by a language model alone—these problems often require the agent to manipulate a machine environment with tools: reading input files, executing code, and writing outputs to specific files in the required format.
<br><br>
The scores in this category are aggregated from three distinct benchmarks, each targeting different facets of scientific coding and execution. Together, these benchmarks evaluate whether an agent can function as a hands-on scientific assistant—not just by reasoning about code, but by running it in real-world contexts.
<br><br>
For detailed results, use the links above to explore individual benchmark pages.
<br>
"""
DATA_ANALYSIS_DESCRIPTION = """
The **Data Analysis** category evaluates agents on their ability to analyze structured datasets and generate meaningful scientific hypotheses. It currently includes a single benchmark, DiscoveryBench, so the category-level scores are the same as the benchmark-level results.
<br><br>
As additional benchmarks are added in the future, this category will expand to cover a broader range of data-driven reasoning tasks across scientific domains.
<br>
"""
DISCOVERY_DESCRIPTION = """
The **End-to-End Discovery** category tests whether agents can carry out a complete scientific workflow, from task description to experiment design, code execution, results  analysis, and report writing. These tasks require agents to integrate multiple capabilities, producing not just answers but full research artifacts.
<br><br>
Scores in this category are aggregated from two benchmarks, providing the first standardized way to evaluate automated scientific discovery (ASD) agents across all stages of the research process. Use the links above to explore individual benchmark pages.
<br>
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
#intro-paragraph {
    font-size: 18px;
    max-width: 60%;
    padding-left: 25px;
}
#about-content {
    font-size: 18px;
    max-width: 60%;
    padding-left: 25px;
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
#page-content-wrapper{
    padding-left: 25px;
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
        overflow: visible !important;
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
.sub-nav-label {
    font-weight: bold;
    font-size: 16px;
    display: flex;
    align-items: center;
}
.wrap-header-df th span{
    white-space: normal !important;
    word-break: normal !important;
    overflow-wrap: break-word !important;
    line-height: 1.2 !important;
    vertical-align: top !important;
    font-size: 12px !important;
    font-family: 'Manrope';
}
.wrap-header-df th {
    height: auto !important;
}
.wrap-header-df .cell-wrap img {
    width: 16px;
    height: 16px;
    vertical-align: middle;
}
#legend-markdown img {
    width: 16px;
    height: 16px;
    vertical-align: middle;
}
/*------ Global tooltip styles ------*/
.tooltip-icon {
    display: inline-block;
    cursor: help;
    position: relative;
}
.tooltip-icon::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 125%;
    background-color: #105257;
    color: #fff;
    padding: 10px;
    border-radius: 4px;
    font-size: 12px;
    opacity: 0;
    transition: opacity 0.2s;
    white-space: pre-line;
    width: max-content;
    text-align: left;
    pointer-events: none;
    max-width: 300px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 1000;
}
@media (max-width: 768px) {
    .tooltip-icon::after {
        max-width: 250px;
    }
}
.tooltip-icon:hover::after {
    opacity: 1;
}
/*------ Openness label tooltip styles ------*/
.styler,
#openness-label-html,
#agent-tooling-label-html {
    overflow: visible !important;
}
/*------ Table cell tooltip styles ------*/
.wrap.default.full,
span.wrap[tabindex="0"][role="button"][data-editable="false"] {
  overflow: visible !important;
}

.cell-tooltip-icon::after {
    height: fit-content;
    top: 125%;
}
/*------ Table column description tooltip styles ------*/
#legend-markdown,
#leaderboard-accordion {
    overflow: visible !important;
}

/* --- inside table tooltips --- */
.native-tooltip-icon {
    cursor: help;
    text-decoration: underline dotted 1px;
}
/* Main Nav bar styling */
.nav-holder nav {
    display: grid !important;
    grid-template-columns: auto auto auto auto 1fr auto auto !important;
    gap: 10px 20px !important; /* Vertical and horizontal spacing */
    width: 100% !important;
    align-items: center;
}
.nav-holder nav a[href*="about"] {
    grid-row: 1 !important;
    grid-column: 6 !important;
}
.nav-holder nav a[href*="submit"] {
    grid-row: 1 !important;
    grid-column: 7 !important;
}
.nav-holder nav a[href*="literature-understanding"] {
    grid-row: 3 !important;
    grid-column: 1 !important;
    width: fit-content !important;
    justify-self: center !important;
}
.nav-holder nav a[href*="code-execution"] {
    grid-row: 3 !important;
    grid-column: 2 !important;
    padding-right: 20px !important;
    justify-self: center !important; 
}
.nav-holder nav a[href*="data-analysis"] {
    grid-row: 3 !important;
    grid-column: 3 !important;
    padding-right: 20px !important;
    justify-self: center !important;
}
.nav-holder nav a[href*="discovery"] {
    grid-row: 3 !important;
    grid-column: 4 !important;
    padding-right: 20px !important;
    justify-self: center !important;
}
.nav-holder nav::after {
    content: ''; /* Required for pseudo-elements to appear */
    background-color: #C9C9C3;
    height: 1px; 
    grid-row: 2 !important;
    grid-column: 1 / -1 !important;
}
.benchmark-header {
    display: flex !important;
    align-items: center !important;
    gap: 20px !important;
    width: 100% !important;
}
.scroll-up-container .prose {
    display: flex;
    justify-content: flex-end; 
}
.scroll-up-button {
    flex-grow: 0;
    display: flex;
    color: #032629;
    background-color: #faf2e9;
    align-items: flex-end;
    height: 57px;
    padding: 0px;
    padding-bottom: 2px;
    min-width: 50px;
}
.dark .scroll-up-button {
    color: #faf2e9;
    background-color: #032629;
}
/*------ Submission Page CSS ------*/
#custom-form-group {
    border: 1px solid #000 !important; 
    border-radius: 4px !important;
    padding: 16px 16px 0px 0px !important;
    overflow: visible !important;    
}

#openness-label-html,
#agent-tooling-label-html {
    padding-left: 12px;
}

#custom-form-group fieldset {
    padding-top: 0px !important;
}

#agent-tooling-label-html {
    padding-top: 6px;
}

#custom-form-group,
.styler {
    background: none;
}

#feedback-button {
    display: inline-block;
    background-color: #345d60;
    color: white;
    border: none;
    border-radius: 4px;
    margin: 5px 0 0 24px;
    padding: 15px 20px;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.3s ease;
    text-decoration: none;
}

#feedback-button:hover {
    background-color: #5d888b;
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.3);
}
.dark #main-header h2 {
    color: #0fcb8c; 
}
#main-header h2 {
    color: #f0529c;
}

/* --- New HTML-Based Tooltip Styles --- */
.tooltip-icon-legend {
    position: relative;
    cursor: help;
    display: inline-block;
}

/* The HTML pop-up card.*/
.tooltip-card {
    /* Hiding mechanism */
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.2s;
    pointer-events: none;

    /* Card appearance */
    position: absolute;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    z-index: 1000;
    background-color: #083c40;
    color: #e5e7eb;
    border-radius: 12px;
    padding: 15px;
    width: max-content;
    max-width: 400px;
    text-align: left;
}

.tooltip-icon-legend:hover .tooltip-card {
    opacity: 1;
    visibility: visible;
}

.tooltip-card h3 {
    font-size: 18px; 
    color: #fff; 
    margin-top: 0; 
    margin-bottom: 12px;
}
.tooltip-card .tooltip-description {
    margin-bottom: 20px; 
    line-height: 1.3;
}
.tooltip-card .tooltip-items-container {
    display: flex; 
    flex-direction: column; 
    gap: 10px;
}
.tooltip-card .tooltip-legend-item {
    display: flex; 
    align-items: 
    flex-start; 
    gap: 10px;
}
.tooltip-card .tooltip-legend-item img {
    width: 20px; 
    height: 20px; 
    margin-top: 2px;
}
.tooltip-card .tooltip-legend-item div {
    display: flex; 
    flex-direction: column;
}
.tooltip-card .tooltip-legend-item strong {
    font-weight: 600; 
    color: #fff;
}
.tooltip-card .tooltip-legend-item span {
    font-size: 13px; 
    line-height: 1.3;
}
.tooltip-sub-list {
    list-style-type: '• '; 
    padding-left: 18px;         
    font-size: 13px;
    line-height: 1.3;  
    display: flex;
    flex-direction: column;   
}       
/* About Page CSS */
#about-page-content-wrapper {
    margin-left: auto;
    margin-right: auto;
    max-width: 800px; 
    padding: 0 24px;
    display: flex;
    flex-direction: column; 
    gap: 40px; 
    margin-top: 40px;
    opacity: 85%; 
}
#leaderboard-accordion table {
    width: auto !important;
    margin-right: auto !important;
}
.info-list {
    padding-left: 20px;
}
"""
