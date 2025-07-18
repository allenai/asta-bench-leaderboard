import gradio as gr
import pandas as pd

# Import our UI factories and the data loader
from ui_components import create_leaderboard_display, create_benchmark_details_display, get_full_leaderboard_data, create_sub_navigation_bar
from content import LIT_DESCRIPTION
# Define the category for this page
CATEGORY_NAME = "Literature Understanding"

with gr.Blocks() as demo:
    gr.Markdown(f"## Astabench{CATEGORY_NAME} Leaderboard")

    validation_df, validation_tag_map = get_full_leaderboard_data("validation")
    test_df, test_tag_map = get_full_leaderboard_data("test")
    gr.Markdown(LIT_DESCRIPTION, elem_id="category-intro")
    with gr.Column(elem_id="validation_nav_container", visible=False) as validation_nav_container:
        create_sub_navigation_bar(validation_tag_map, CATEGORY_NAME)

    with gr.Column(elem_id="test_nav_container", visible=True) as test_nav_container:
        create_sub_navigation_bar(test_tag_map, CATEGORY_NAME)

    # --- This page now has two main sections: Validation and Test ---
    with gr.Tabs():
        with gr.Tab("Results: Test Set") as test_tab:
            # Repeat the process for the "test" split
            test_df, test_tag_map = get_full_leaderboard_data("test")

            if not test_df.empty:
                create_leaderboard_display(
                    full_df=test_df,
                    tag_map=test_tag_map,
                    category_name=CATEGORY_NAME,
                    split_name="test"
                )
                create_benchmark_details_display(
                    full_df=test_df,
                    tag_map=test_tag_map,
                    category_name=CATEGORY_NAME
                )
            else:
                gr.Markdown("No data available for test split.")
        with gr.Tab("Results: Validation Set") as validation_tab:
            # 1. Load all necessary data for the "validation" split ONCE.
            validation_df, validation_tag_map = get_full_leaderboard_data("validation")

            if not validation_df.empty:
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
                    category_name=CATEGORY_NAME
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