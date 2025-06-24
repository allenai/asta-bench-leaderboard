import gradio as gr

# Import our new UI factory
from ui_components import create_leaderboard_display

with gr.Blocks() as demo:
    gr.Markdown("## Data Analysis Results")
    # This page shows the "Literature Understanding" view.

    with gr.Tabs():
        with gr.Tab("Results: Validation"):
            # Call the factory with the "Literature Understanding" tag
            create_leaderboard_display(split_name="validation", tag_name="Data Analysis")

        with gr.Tab("Results: Test"):
            create_leaderboard_display(split_name="test", tag_name="Data Analysis")