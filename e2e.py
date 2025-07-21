import gradio as gr
from content import DISCOVERY_DESCRIPTION
from category_page_builder import build_category_page
# Define the category for this page
CATEGORY_NAME = "Discovery"

with gr.Blocks() as demo:
    gr.Markdown(f"## Astabench{CATEGORY_NAME} Leaderboard")
    build_category_page(CATEGORY_NAME, DISCOVERY_DESCRIPTION)