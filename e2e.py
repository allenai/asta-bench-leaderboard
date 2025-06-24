import gradio as gr
import pandas as pd

# Import our UI factories and the data loader
from ui_components import create_leaderboard_display, create_benchmark_details_display, get_full_leaderboard_data

# Define the category for this page
CATEGORY_NAME = "Discovery"

with gr.Blocks() as demo:
    gr.Markdown(f"## {CATEGORY_NAME} Leaderboard Results")

    # --- This page now has two main sections: Validation and Test ---
    with gr.Tabs():
        with gr.Tab("Results: Validation"):
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

        with gr.Tab("Results: Test"):
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