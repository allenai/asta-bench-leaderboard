import gradio as gr
from main_page import load_display_data_for_split


with gr.Blocks() as demo:
    gr.Markdown("# Literature Understanding")
    gr.Markdown("This is some text to explain what literature Understanding metrics are!")
    t = gr.Textbox()
    demo.load(lambda : "Loaded", None, t)
    with gr.Row():
        initial_choices_for_dropdown = ["Overall", "Literature Understanding"]
        global_tag_dropdown = gr.Dropdown(
            choices=initial_choices_for_dropdown,
            value="Overall",
            label="Filter by Tag/Capability",
            elem_id="global_tag_dropdown_lit" # Note: elem_id should be unique if multiple instances exist
        )
    # Tabs for Test and Validation
    with gr.Tab("Results: Validation") as val_tab:
        val_split_name = gr.State("validation")
        with gr.Row():
            val_refresh_button = gr.Button("Refresh Validation Tab")
        val_df_output = gr.DataFrame(interactive=False, wrap=True, label="Validation Leaderboard")
        val_scatter_plot_output = gr.Plot(label="Score vs. Cost (Validation)")

        initial_val_df, initial_val_scatter, _ = load_display_data_for_split(
            "validation", global_tag_dropdown.value
        )
        val_df_output.value = initial_val_df
        val_scatter_plot_output.value = initial_val_scatter

        val_refresh_button.click(
            fn=load_display_data_for_split,
            inputs=[val_split_name, global_tag_dropdown],
            outputs=[val_df_output, val_scatter_plot_output, global_tag_dropdown]
        )

    with gr.Tab("Results: Test") as test_tab:
        test_split_name = gr.State("test")
        with gr.Row():
            test_refresh_button = gr.Button("Refresh Test Tab")
        test_df_output = gr.DataFrame(interactive=False, wrap=True, label="Test Leaderboard")

        test_scatter_plot_output = gr.Plot(label="Score vs. Cost (Test)")


        initial_test_df, initial_test_scatter, _ = load_display_data_for_split(
            "test", global_tag_dropdown.value
        )
        test_df_output.value = initial_test_df
        test_scatter_plot_output.value = initial_test_scatter

        test_refresh_button.click(
            fn=load_display_data_for_split,
            inputs=[test_split_name, global_tag_dropdown],
            outputs=[test_df_output, test_scatter_plot_output, global_tag_dropdown]
        )

if __name__ == "__main__":
    demo.launch()