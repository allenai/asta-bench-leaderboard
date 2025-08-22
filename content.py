import re

def create_gradio_anchor_id(text: str, validation) -> str:
    """
    Replicates the ID format created by gr.Markdown(header_links=True).
    Example: "Paper Finder Validation" -> "h-paper-finder-validation"
    """
    text = text.lower()
    text = re.sub(r'\s+', '-', text) # Replace spaces with hyphens
    text = re.sub(r'[^\w-]', '', text) # Remove non-word characters
    if validation:
        return f"h-{text}-leaderboard-1"
    return f"h-{text}-leaderboard"


TITLE = """<h1 align="left" id="space-title">AstaBench Leaderboard</h1>"""

INTRO_PARAGRAPH = """
<p>
    <strong>AstaBench</strong> provides an aggregated view of agent performance and efficiency across all benchmarks in all four categories. We report:
</p>

<ul class="info-list">
    <li>
        <strong>Overall score:</strong> A macro-average of the four category-level average scores. Each category contributes equally, regardless of how many benchmarks it includes. This ensures fair comparisons across agents with different domain strengths.
    </li>
    <li>
        <strong>Overall cost:</strong> A macro-average of the agent’s cost per problem across all categories, in USD. Each category contributes equally.
    </li>
</ul>

<p>
    This view is designed for quick comparison of general-purpose scientific agents. For more details on how we calculate scores and cost, please see the <a href="/about" style="color: #0FCB8C; text-decoration: underline;">About</a> Page.
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
SUBMISSION_CONFIRMATION = """
**Your agent has been submitted to AstaBench for evaluation.**
<br><br>
🙏 Thanks for contributing!
<br><br>
You'll receive an email shortly with confirmation and next steps. If there are any issues with your submission, our team will reach out within 5–7 business days.
<br><br>
We appreciate your support in advancing scientific AI.
"""

# External URLs for benchmark descriptions
SCHOLAR_QA_CS_URL = "https://www.semanticscholar.org/paper/OpenScholar%3A-Synthesizing-Scientific-Literature-LMs-Asai-He/b40df4b273f255b3cb5639e220c8ab7b1bdb313e"
LITQA2_URL = "https://www.semanticscholar.org/paper/Language-agents-achieve-superhuman-synthesis-of-Skarlinski-Cox/fa5f9aa1cb6f97654ca8e6d279ceee1427a87e68"
ARXIV_DIGESTABLES_URL = "https://www.semanticscholar.org/paper/ArxivDIGESTables%3A-Synthesizing-Scientific-into-Newman-Lee/c7face35e84f2cb04fb1600d54298799aa0ed189"
SUPER_URL = "https://www.semanticscholar.org/paper/SUPER%3A-Evaluating-Agents-on-Setting-Up-and-Tasks-Bogin-Yang/053ef8299988680d47df36224bfccffc817472f1"
CORE_BENCH_URL = "https://www.semanticscholar.org/paper/CORE-Bench%3A-Fostering-the-Credibility-of-Published-Siegel-Kapoor/4c913d59d150fe7581386b87dfd9f90448a9adee"
DS1000_URL = "https://arxiv.org/abs/2211.11501"
DISCOVERY_BENCH_URL = "https://www.semanticscholar.org/paper/DiscoveryBench%3A-Towards-Data-Driven-Discovery-with-Majumder-Surana/48c83799530dc523ee01e6c1c40ad577d5c10a16"

# Helper function to create external links
def external_link(url, text, is_s2_url=False):
    url = f"{url}?utm_source=asta_leaderboard" if is_s2_url else url
    return f"<a href='{url}' target='_blank' rel='noopener noreferrer'>{text}</a>"

def internal_leaderboard_link(text, validation):
    anchor_id = create_gradio_anchor_id(text, validation)
    return f"<a href='#{anchor_id}'>{text}</a>"

# Function to get benchmark descriptions with validation flag
def get_benchmark_description(benchmark_name, validation):
    descriptions = {
    'PaperFindingBench': (
        "PaperFindingBench assesses an agent's ability to locate sets of papers based on a natural language "
        "description that may involve both the papers' content and metadata, such as the author or publication year."
    ),
    'LitQA2-FullText-Search': (
        f"A version of {internal_leaderboard_link('LitQA2-FullText', validation)} that isolates the retrieval aspect of the task. "
        f"This benchmark features the same multi-choice questions as {internal_leaderboard_link('LitQA2-FullText', validation)}, but the agent is not evaluated on answering the actual question "
        "but rather on providing a ranked list of papers in which the answer is likely to be found."
    ),
    'ScholarQA-CS2': (
        "ScholarQA-CS2 assesses long-form model responses to literature review questions in the domain of computer science. "
        "Answers are expected to be comprehensive reports, such as those produced by deep research systems. "
        f"This benchmark advances on the previously released {external_link(SCHOLAR_QA_CS_URL, 'ScholarQA-CS', is_s2_url=True)} "
        "by using queries from real-world usage, and introducing new evaluation methods for coverage and precision "
        "of both the report text and its citations."
    ),
    'LitQA2-FullText': (
        f"{external_link(LITQA2_URL, 'LitQA2', is_s2_url=True)}, a benchmark introduced by FutureHouse, gauges a model's ability to answer questions that require document retrieval from the scientific literature. "
        "It consists of multiple-choice questions that necessitate finding a unique paper and analyzing its detailed full text to spot precise information; these questions cannot be answered from a paper’s abstract. "
        "While the original version of the benchmark provided for each question the title of the paper in which the answer can be found, it did not specify the overall collection to search over. In our version, "
        "we search over the index we provide as part of the Asta standard toolset. The “-FullText” suffix indicates we consider only the subset of LitQA2 questions for which "
        "the full-text version of the answering paper is open source and available in our index."
    ),
    'ArxivDIGESTables-Clean': (
        f"{external_link(ARXIV_DIGESTABLES_URL, 'ArxivDIGESTables', is_s2_url=True)} assesses the ability of models to construct literature review tables, i.e., tables whose rows are papers and whose columns constitute a set of "
        "aspects used to compare and contrast the papers. The goal is to construct such tables given a set of related papers and a table caption describing the user's goal. Generated tables are evaluated by "
        "comparing them to actual tables published in ArXiv papers. The “-Clean” suffix indicates a curated subset of ArxivDIGESTables which drops tables that are either trivial or impossible to reconstruct from full-texts."
    ),
    'SUPER-Expert': (
        "SUPER-Expert evaluates the capability of models in setting up and executing tasks from low-resource "
        "research repositories—centralized databases containing research data and related materials. "
        f"The \"-Expert\" split indicates the name of the most challenging split in the {external_link(SUPER_URL, 'original SUPER benchmark', is_s2_url=True)} "
        "that involves solving reproduction tasks from scratch and without any intermediate hints or details "
        "about the important landmarks involved in each task."
    ),
    'CORE-Bench-Hard': (
        "Core-Bench-Hard tests computational reproducibility, a task involving reproducing the results of a study "
        "using provided code and data. It consists of both language-only and vision-language challenges across "
        "multiple difficulty levels. "
        f"The \"-Hard\" split refers to the name of the most challenging split in the original {external_link(CORE_BENCH_URL, 'Core-bench benchmark', is_s2_url=True)} "
        "where only a README file is provided with no instructions or an auxiliary Dockerfile."
    ),
    'DS-1000': (
        "DS-1000 is an established code generation benchmark containing Python data science coding questions "
        "originally sourced from StackOverflow. It's designed to reflect an array of diverse, realistic, and "
        "practical use cases and directly involves many of the Python libraries commonly used in data science "
        f"and machine learning research. We split the original {external_link(DS1000_URL, 'dataset')} "
        "into 100 validation and 900 test problems."
    ),
    'DiscoveryBench': (
        "DiscoveryBench is the first comprehensive benchmark to formalize the multi-step process of data-driven "
        "analysis and discovery (i.e., data loading, transformation, statistical analysis, and modeling). "
        f"Originally introduced {external_link(DISCOVERY_BENCH_URL, 'here', is_s2_url=True)}, it is designed to systematically "
        "evaluate how well current LLMs can replicate or reproduce published scientific findings across diverse "
        "domains, including social science, biology, history, and more."
    ),
    'E2E-Bench': (
        "E2E-Bench is the \"decathlon\" of AI-assisted research. It measures whether a system can run the entire "
        "research pipeline, starting with an initial task description, to designing and performing (software) "
        "experiments, to analyzing and writing up the results."
    ),
    'E2E-Bench-Hard': (
        f"E2E-Bench-Hard is a more challenging variant of {internal_leaderboard_link('E2E-Bench', validation)}. Tasks are generated using the HypER system, "
        "which identifies research trends and proposes new, underexplored problems. Unlike the regular version, "
        "these tasks are not simplified or curated for accessibility; they are reviewed only for feasibility. "
        "This version is intended to test whether systems can handle more complex and less-structured research "
        f"scenarios, following the same end-to-end process as {internal_leaderboard_link('E2E-Bench', validation)}."
    )
    }
    
    return descriptions.get(benchmark_name, "")

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
/* CSS Color Variables using Gradio theme */
:root {
    --color-primary-green: var(--primary-900); /* #0FCB8C */
    --color-primary-pink: var(--secondary-900); /* #f0529c */
    --color-neutral-light: var(--neutral-200); /* #C9C9C3 */
    --color-background-light: var(--neutral-50); /* #FAF2E9 */
    --color-background-dark: var(--neutral-900); /* #032629 */
    --color-text-light: var(--neutral-50); /* #FAF2E9 */
}

/* Global Styles */
h2 {
    overflow: hidden;
}

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
.sub-nav-bar-container {
    display: flex !important;
    flex-wrap: nowrap !important; 
    align-items: center !important; 
    gap: 20px !important;
}
.dark .primary-link-button {
    color: var(--color-primary-green);
}
.primary-link-button {
    background: none;
    border: none;
    padding: 0;
    margin: 0;
    font-family: inherit;
    font-size: 16px;
    color: var(--color-primary-pink);
    text-decoration: none;
    cursor: pointer;
    white-space: nowrap;
}
.primary-link-button:hover {
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
.benchmark-main-subtitle{
    color: var(--color-primary-green);
    overflow: hidden;
    padding-top: 120px;
}
.benchmark-title{
    color: var(--color-primary-pink);
    margin-top: 50px;
        font-size: 20px;
}
.dark .benchmark-title{
    color: var(--color-primary-green);
}
.benchmark-description {
    margin: 20px 0;
    max-width: 800px;
}
/*------ Submission Page CSS ------*/
#submission-modal .modal-container {
    height: auto;
    max-width: 600px;
}

#submission-modal-content {
    padding: 20px;
    background-color: inherit;
    border-radius: 8px;
    text-align: center;
}

#submission-modal-content p{
    font-size: 16px;
}

.spinner-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 30px;
}
    
.spinner {
    width: 50px;
    height: 50px;
    border: 5px solid #dee2e6;
    border-top: 5px solid #007bff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 20px;
}
    
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
    
#submission-page-container {
    max-width: 800px;
    margin: 0 auto;
}

#submission-file-label {
    padding: 10px;
}

#submission-button {
    max-width: fit-content;
    font-size: 14px;
}

.custom-form-group {
    border: 1px solid #000 !important; 
    border-radius: 4px !important;
    padding: 24px !important;
    overflow: visible !important;    
}

#openness-label-html,
#agent-tooling-label-html,
#agent-info-label-html,
#submitter-info-label-html,
#username-label-html,
#email-label-html,
#role-label-html  {
    padding-left: 12px;
}

.form-label {
    margin: 4px 0px 0px 6px;
}

.form-label-fieldset {
    padding-top: 10px !important;
}

#agent-tooling-label-html {
    padding-top: 6px;
}

.custom-form-group,
.styler {
    background: none;
}

#feedback-button {
    display: inline-block;
    background-color: #345d60;
    color: white;
    border: none;
    border-radius: 4px;
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

/* The HTML pop-up card tooltips.*/
.tooltip-card {
    /* Hiding mechanism */
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.2s;
    pointer-events: none;
    /* Card appearance */
    position: fixed;
    z-index: 1000;
    background-color: #083c40;
    color: #e5e7eb;
    border-radius: 12px;
    padding: 15px;
    width: max-content;
    max-width: 400px;
    text-align: left;
}
.tooltip-card.visible {
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
.table-legend-item {  
    display: flex; 
    align-items: center; 
    white-space: nowrap; 
    margin-top: 8px; 
    flex-wrap: wrap;
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
    margin-bottom: 60px;
}
.link-buttons-container {
    display: flex;
    flex-wrap: wrap; /* Allows buttons to stack on very narrow screens */
    gap: 16px;     
    margin-top: 16px;
}
.link-button {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-grow: 1; 
    background-color: #083c40; 
    padding: 16px 20px;
    font-weight: 600;
    border-radius: 12px;
    text-decoration: none; 
    transition: background-color 0.2s ease-in-out;
}
.link-button:hover {
    background-color: #0a4c52; 
}
.external-link-icon {
    font-size: 20px;
    line-height: 1;
    margin-left: 12px;
}

#leaderboard-accordion table {
    width: auto !important;
    margin-right: auto !important;
}
.info-list {
    padding-left: 20px;
}

/* Smooth scrolling for the entire page */
html {
    scroll-behavior: smooth;
}
/* Home Page Styling */
.diagram-placeholder {
    width: 100%;
    height: 100%; 
    min-height: 250px; 
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #FAF2E9; 
    color: #F0529C;          
    border-radius: 8px;
    font-size: 14px;
    text-align: center;
}
/* 2. Responsive behavior for smaller screens */
@media (max-width: 900px) {
    #intro-row {
        flex-direction: column;
    }
}
#home-page-content-wrapper{
    margin-top: 40px;
}
#intro-paragraph {
    max-width: 90%;
}
/* Plot legend styles */
.plot-legend-container {
    min-height: 572px;
    background-color: #fff;
    padding: 24px 32px;
    border: 1px solid black;
    border-radius: 4px;
}

.dark .plot-legend-container {
    background: rgba(250, 242, 233, 0.1);
    border-color: rgb(159, 234, 209);
}

#plot-legend-logo {
    margin-bottom: 24px;
}

#plot-legend-logo img {
    height: 19px;
}

.plot-legend-category-heading {
    font-size: 16px;
    font-weight: 700;    
}

.plot-legend-item {
    display: flex;      
    margin-top: 8px;
}


.plot-legend-item-text .description {
    color: #888;
    font-size: 12px;
}

.plot-legend-item-svg {
    margin-top: 3px;
    width: 14px;
    height: 14px;
    margin-right: 8px;
}

.plot-legend-tooling-svg {
    height: 16px;
    width: 16px;
    margin-top: 2px;
}

#plot-legend-item-pareto-svg {
    width: 18px;
    height: 18px;
    margin-right: 2px;
}
"""
