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

def build_page():
    with gr.Column(elem_id="home-page-content-wrapper"):
        with gr.Row(elem_id="intro-row"):
            with gr.Column(scale=6):
                gr.HTML(INTRO_PARAGRAPH, elem_id="intro-paragraph")

    # --- Leaderboard Display Section ---
    gr.Markdown("---")
    CATEGORY_NAME = "Overall"
    gr.HTML(f'<h2>AstaBench {CATEGORY_NAME} Leaderboard <span style="font-weight: normal; color: inherit;">(Aggregate)</span></h2>', elem_id="main-header")

    with gr.Tabs() as tabs:
        with gr.Tab("Results: Test Set") as test_tab:
            test_df, test_tag_map = get_full_leaderboard_data("test")
            if not test_df.empty:
                gr.Markdown("**Test Set** results are reserved for final assessment. This helps ensure that the agent generalizes well to unseen problems.")
                create_leaderboard_display(
                    full_df=test_df,
                    tag_map=test_tag_map,
                    category_name=CATEGORY_NAME, # Use our constant
                    split_name="test"
                )
            else:
                gr.Markdown("No data available for test split.")
        with gr.Tab("Results: Validation Set") as validation_tab:
            # 1. Load all necessary data for the "validation" split ONCE.
            validation_df, validation_tag_map = get_full_leaderboard_data("validation")
            # Check if data was loaded successfully before trying to display it
            if not validation_df.empty:
                gr.Markdown("**Validation Set** results are used during development to tune and compare agents before final testing.")
                # 2. Render the display by calling the factory with the loaded data.
                create_leaderboard_display(
                    full_df=validation_df,
                    tag_map=validation_tag_map,
                    category_name=CATEGORY_NAME, # Use our constant
                    split_name="validation"
                )
            else:
                gr.Markdown("No data available for validation split.")

    # hiding this for now till we have the real paper data
    # with gr.Accordion("📙 Citation", open=False):
    #     gr.Textbox(value=CITATION_BUTTON_TEXT, label=CITATION_BUTTON_LABEL, elem_id="citation-button-main", interactive=False)


    # JavaScript to show the TEST nav, hide the VALIDATION nav, AND fix the plots.
    show_validation_js = """
        () => {setTimeout(() => { window.dispatchEvent(new Event('resize')) }, 0);}
        """
    # Assign the pure JS functions to the select events. No Python `fn` is needed.
    validation_tab.select(fn=None, inputs=None, outputs=None, js=show_validation_js)

if __name__ == "__main__":
    demo.launch()