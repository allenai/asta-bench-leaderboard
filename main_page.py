import matplotlib
matplotlib.use('Agg')
import gradio as gr


from ui_components import create_leaderboard_display, get_full_leaderboard_data

from content import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    INTRO_PARAGRAPH
)

# --- Global State for Viewers (simple caching) ---
CACHED_VIEWERS = {}
CACHED_TAG_MAPS = {}

with gr.Blocks(fill_width=True) as demo:
    gr.Markdown(INTRO_PARAGRAPH, elem_id="intro-paragraph")
    # --- Leaderboard Display Section ---
    gr.Markdown("---")
    CATEGORY_NAME = "Overall"
    gr.Markdown(f"## {CATEGORY_NAME} Categories Aggregated")

    with gr.Tabs() as tabs:
        with gr.Tab("Results: Validation"):
            # 1. Load all necessary data for the "validation" split ONCE.
            validation_df, validation_tag_map = get_full_leaderboard_data("validation")

            # Check if data was loaded successfully before trying to display it
            if not validation_df.empty:
                # 2. Render the display by calling the factory with the loaded data.
                create_leaderboard_display(
                    full_df=validation_df,
                    tag_map=validation_tag_map,
                    category_name=CATEGORY_NAME, # Use our constant
                    split_name="validation"
                )
            else:
                gr.Markdown("No data available for validation split.")

        with gr.Tab("Results: Test"):
            test_df, test_tag_map = get_full_leaderboard_data("test")
            if not test_df.empty:
                create_leaderboard_display(
                    full_df=test_df,
                    tag_map=test_tag_map,
                    category_name=CATEGORY_NAME, # Use our constant
                    split_name="test"
                )
            else:
                gr.Markdown("No data available for test split.")

    with gr.Accordion("📙 Citation", open=False):
        gr.Textbox(value=CITATION_BUTTON_TEXT, label=CITATION_BUTTON_LABEL, elem_id="citation-button-main", interactive=False)


if __name__ == "__main__":
    demo.launch()