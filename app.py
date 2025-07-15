# app.py
import gradio as gr
import os

from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import HfApi
import literature_understanding, main_page, c_and_e, data_analysis, e2e, submission

from content import css

# --- Constants and Configuration  ---
LOCAL_DEBUG = not (os.environ.get("system") == "spaces")
IS_INTERNAL = os.environ.get("IS_INTERNAL", "false").lower() == "true"
OWNER = "allenai"
PROJECT_NAME = "asta-bench" + ("-internal" if IS_INTERNAL else "")
LEADERBOARD_PATH = f"{OWNER}/{PROJECT_NAME}-leaderboard"
api = HfApi()
LOGO_PATH = "assets/logo.svg"



# --- Theme Definition ---
theme = gr.themes.Base(
    primary_hue=gr.themes.Color(c100="#CFF5E8", c200="#B7EFDD", c300="#9FEAD1", c400="#87E5C5", c50="#E7FAF3", c500="#6FE0BA", c600="#57DBAF", c700="#3FD5A3", c800="#27D09C", c900="#0FCB8C", c950="#0fcb8c"),
    secondary_hue=gr.themes.Color(c100="#FCDCEB", c200="#FBCBE1", c300="#F9BAD7", c400="#F7A8CD", c50="#FDEEF5", c500="#F697C4", c600="#F586BA", c700="#F375B0", c800="#F263A6", c900="#F0529C", c950="#F0529C"),
    neutral_hue=gr.themes.Color(c100="#FDF9F4", c200="#C9C9C3", c300="#B0B5AF", c400="#97A09C", c50="#FAF2E9", c500="#7F8C89", c600="#667876", c700="#344F4F", c800="#1C3A3C", c900="#032629", c950="032629"),
    font=[gr.themes.GoogleFont('Manrope'), 'ui-sans-serif', 'sans-serif', 'sans-serif'],
    font_mono=[gr.themes.GoogleFont('Roboto Mono'), 'ui-monospace', 'monospace', 'monospace'],
).set(
    body_text_color='*neutral_950',
    body_text_color_dark='*neutral_50',
    background_fill_primary='*neutral_50',
    background_fill_primary_dark='*neutral_900',
    background_fill_secondary='*neutral_100',
    background_fill_secondary_dark='*neutral_800',
    border_color_accent='*secondary_900',
    border_color_accent_subdued='*neutral_400',
    border_color_accent_subdued_dark='*neutral_400',
    color_accent='*primary_900',
    color_accent_soft='*neutral_200',
    color_accent_soft_dark='*neutral_800',
    link_text_color='*secondary_900',
    link_text_color_dark='*primary_900',
    link_text_color_active_dark='*primary_600',
    link_text_color_hover_dark='*primary_700',
    link_text_color_visited_dark='*primary_600',
    table_even_background_fill='*neutral_100',
    table_even_background_fill_dark='*neutral_800',
    button_primary_background_fill='*secondary_900',
    button_primary_background_fill_dark='*primary_900',
    button_primary_background_fill_hover='*secondary_600',
    button_primary_background_fill_hover_dark='*primary_600',
    button_secondary_background_fill="#9FEAD1",
    button_secondary_background_fill_dark="#9FEAD1",
    button_secondary_text_color="*neutral_900",
    button_secondary_text_color_dark="*neutral_900",
    block_title_text_color="*neutral_900",
    button_primary_text_color='*neutral_900',
    block_title_text_color_dark="#ffffff",
    checkbox_label_text_color_dark="#000",
    button_primary_text_color_dark='*neutral_900',
    block_border_color="#032629",
    block_border_color_dark="#9fead1",
    block_background_fill_dark="#032629",
    block_background_fill="#FAF2E9",
)
def render_logo():
    return gr.Image(
        value=LOGO_PATH,
        show_label=False,
        interactive=False,
        container=False,
        show_download_button=False,
        show_fullscreen_button=False,
        elem_id="logo-image"
    )
# --- Gradio App Definition ---
demo = gr.Blocks(theme=theme, css=css)
with demo:
    render_logo()
    main_page.demo.render()
with demo.route("Literature Understanding", "/literature-understanding"):
    render_logo()
    literature_understanding.demo.render()
with demo.route("Code & Execution", "/code-execution"):
    render_logo()
    c_and_e.demo.render()
with demo.route("Data Analysis", "/data-analysis"):
    render_logo()
    data_analysis.demo.render()
with demo.route("Discovery", "/discovery"):
    render_logo()
    e2e.demo.render()
with demo.route(" 🚀 Submit an Agent"):
    render_logo()
    submission.demo.render()

# --- Scheduler and Launch
def restart_space_job():
    print("Scheduler: Attempting to restart space.")
    try:
        api.restart_space(repo_id=LEADERBOARD_PATH)
        print("Scheduler: Space restart request sent.")
    except Exception as e:
        print(f"Scheduler: Error restarting space: {e}")
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(restart_space_job, "interval", hours=1)
    scheduler.start()


# Launch the Gradio app
if __name__ == "__main__":
    if LOCAL_DEBUG:
        print("Launching in LOCAL_DEBUG mode.")
        def get_initial_global_tag_choices(): return ["Overall", "TagA"]
        demo.launch(debug=True)
    else:
        print("Launching in Space mode.")
        # For Spaces, share=False is typical unless specific tunneling is needed.
        # debug=True can be set to False for a "production" Space.
        demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=False)

