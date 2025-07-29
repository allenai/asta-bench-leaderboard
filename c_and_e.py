import gradio as gr
from content import CODE_EXECUTION_DESCRIPTION
from category_page_builder import build_category_page

# Define the category for this page
CATEGORY_NAME = "Code Execution"

def build_page():
    build_category_page(CATEGORY_NAME, CODE_EXECUTION_DESCRIPTION)