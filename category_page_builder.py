import gradio as gr
import pandas as pd

# Import our UI factories and the data loader
from ui_components import create_leaderboard_display, create_benchmark_details_display, get_full_leaderboard_data, create_sub_navigation_bar
CATEGORY_DIAGRAM_MAP = {
    "Literature Understanding": "assets/literature-understanding.png",
    "Code & Execution": "assets/code-execution.png",
    "Data Analysis": "assets/data-analysis.png",
    "End-to-End Discovery": "assets/end-to-end-discovery.png"
}


def build_category_page(CATEGORY_NAME, PAGE_DESCRIPTION):
    with gr.Column(elem_id="page-content-wrapper"):

        # --- Step 1: Load the data needed for the nav bars FIRST ---
        # This must happen before we start building the UI that uses it.
        validation_df, validation_tag_map = get_full_leaderboard_data("validation")
        test_df, test_tag_map = get_full_leaderboard_data("test")

        # --- Step 2: The Top Section (Title, Description, and Diagram) ---
        with gr.Row(elem_id="intro-row"):

            # --- The Left Column ---
            with gr.Column(scale=1):

                # 1. The main title (as you have it)
                gr.HTML(f'<h2>AstaBench {CATEGORY_NAME} Leaderboard <span style="font-weight: normal; color: inherit;">(Aggregate)</span></h2>', elem_id="main-header")

                # 2. THE FIX: The Benchmark Navigation is now HERE.
                with gr.Column(elem_id="validation_nav_container", visible=False) as validation_nav_container:
                    create_sub_navigation_bar(validation_tag_map, CATEGORY_NAME, validation=True)

                with gr.Column(elem_id="test_nav_container", visible=True) as test_nav_container:
                    create_sub_navigation_bar(test_tag_map, CATEGORY_NAME)

                # 3. The page description now follows the benchmark nav.
                gr.Markdown(PAGE_DESCRIPTION, elem_id="intro-paragraph")

            # --- The Right Column ---
            with gr.Column(scale=1):
                image_path = CATEGORY_DIAGRAM_MAP.get(CATEGORY_NAME)
                if image_path:
                    gr.Image(
                        value=image_path,
                        show_label=False,
                        show_download_button=False,
                        show_fullscreen_button=False,
                        interactive=False,
                        elem_id="diagram-image"
                    )

        # --- This page now has two main sections: Validation and Test ---
        with gr.Tabs():
            with gr.Tab("Results: Test Set") as test_tab:
                # Repeat the process for the "test" split
                if not test_df.empty:
                    gr.Markdown("**Test Set** results are reserved for final assessment. This helps ensure that the agent generalizes well to unseen problems.")
                    create_leaderboard_display(
                        full_df=test_df,
                        tag_map=test_tag_map,
                        category_name=CATEGORY_NAME,
                        split_name="test"
                    )
                    create_benchmark_details_display(
                        full_df=test_df,
                        tag_map=test_tag_map,
                        category_name=CATEGORY_NAME,
                        validation=False,
                    )
                else:
                    gr.Markdown("No data available for test split.")
            with gr.Tab("Results: Validation Set") as validation_tab:
                # 1. Load all necessary data for the "validation" split ONCE.
                if not validation_df.empty:
                    gr.Markdown("**Validation Set** results are used during development to tune and compare agents before final testing.")
                    # 2. Render the main category display using the loaded data.
                    create_leaderboard_display(
                        full_df=validation_df,
                        tag_map=validation_tag_map,
                        category_name=CATEGORY_NAME,
                        split_name="validation"
                    )

                    # 3. Render the detailed breakdown for each benchmark in the category.
                    create_benchmark_details_display(
                        full_df=validation_df,
                        tag_map=validation_tag_map,
                        category_name=CATEGORY_NAME,
                        validation=True,
                    )
                else:
                    gr.Markdown("No data available for validation split.")


        show_validation_js = """
            () => {
                document.getElementById('validation_nav_container').style.display = 'block';
                document.getElementById('test_nav_container').style.display = 'none';
                setTimeout(() => { window.dispatchEvent(new Event('resize')) }, 0);
            }
            """

        # JavaScript to show the TEST nav, hide the VALIDATION nav, AND fix the plots.
        show_test_js = """
            () => {
                document.getElementById('validation_nav_container').style.display = 'none';
                document.getElementById('test_nav_container').style.display = 'block';
            }
            """

        # Assign the pure JS functions to the select events. No Python `fn` is needed.
        validation_tab.select(fn=None, inputs=None, outputs=None, js=show_validation_js)
        test_tab.select(fn=None, inputs=None, outputs=None, js=show_test_js)

    return validation_nav_container, test_nav_container