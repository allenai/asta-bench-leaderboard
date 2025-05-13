TITLE = """<h1 align="center" id="space-title">AstaBench Leaderboard</h1>"""

INTRODUCTION_TEXT = """
<h1>Introduction</h1>
"""

SUBMISSION_TEXT = """
## Submissions
"""
INTRO_PARAGRAPH = "Hello welcome to the AstaBench Leaderboard. We made this leaderboard for you, do you like it? Please say yes, or we will cry. You don't want to make us cry do you? That's what I thought."

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


def hyperlink(link, text):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{text}</a>'


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
.tab-buttons button {
    font-size: 15pt !important;
}

.tab-container {
    overflow: visible !important;  
    padding: 25px 10px 10px 10px;          
    height: auto;                  
}

.markdown-text{font-size: 20pt}
.markdown-text-tiny{font-size: 12pt}
.markdown-text-small{font-size: 16pt}
.markdown-text-large{font-size: 30pt}

#citation-button {
    background-color: #f0f0f0;
}

.label-wrap{
    background-color: #f0f0f0;
    padding: 10px;
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
"""

