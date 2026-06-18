# app.py
import logging

logging.basicConfig(level=logging.WARNING)

import gradio as gr
import urllib.parse

from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import HfApi

from config import LEADERBOARD_PATH, LOCAL_DEBUG
from content import css
from main_page import build_page as build_main_page
from literature_understanding import build_page as build_lit_page
from c_and_e import build_page as build_c_and_e_page
from data_analysis import build_page as build_data_analysis_page
from e2e import build_page as build_e2e_page
from submission import build_page as build_submission_page
from about import build_page as build_about_page

api = HfApi()
LOGO_PATH = "assets/logo.svg"
# JavaScripts
scroll_script = """
<script>
function scroll_to_element(id) {
    console.log("Global scroll_to_element called for ID:", id);
    const element = document.querySelector('#' + id);
    if (element) {
        console.log("Element found:", element);
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
        console.error("Error: Element with ID '" + id + "' not found in the document.");
    }
}
</script>
"""
redirect_script = """
<script>
    if (window.location.pathname === '/') { window.location.replace('/home'); }
</script>
"""
tooltip_script = """
<script>
function initializeSmartTooltips() {
    // Find all tooltip trigger icons
    const tooltipIcons = document.querySelectorAll('.tooltip-icon-legend');

    tooltipIcons.forEach(icon => {
        // Find the tooltip card associated with this icon
        const tooltipCard = icon.querySelector('.tooltip-card');
        if (!tooltipCard) return;

        // Move the card to the end of the <body>. This is the KEY to escaping
        // any parent containers that might clip it.
        document.body.appendChild(tooltipCard);

        // --- MOUSE HOVER EVENT ---
        icon.addEventListener('mouseenter', () => {
            // Get the exact position of the icon on the screen
            const iconRect = icon.getBoundingClientRect();
            // Get the dimensions of the tooltip card
            const cardRect = tooltipCard.getBoundingClientRect();

            // Calculate the ideal top position (above the icon with a 10px gap)
            const top = iconRect.top - cardRect.height - 10;
            
            // --- Smart Centering Logic ---
            // Start by calculating the perfect center
            let left = iconRect.left + (iconRect.width / 2) - (cardRect.width / 2);

            // Check if it's going off the left edge of the screen
            if (left < 10) {
                left = 10; // Pin it to the left with a 10px margin
            }
            // Check if it's going off the right edge of the screen
            if (left + cardRect.width > window.innerWidth) {
                left = window.innerWidth - cardRect.width - 10; // Pin it to the right
            }

            // Apply the calculated position and show the card
            tooltipCard.style.top = `${top}px`;
            tooltipCard.style.left = `${left}px`;
            tooltipCard.classList.add('visible');
        });

        // --- MOUSE LEAVE EVENT ---
        icon.addEventListener('mouseleave', () => {
            // Hide the card
            tooltipCard.classList.remove('visible');
        });
    });
}

// Poll the page until the tooltips exist, then run the initialization.
const tooltipInterval = setInterval(() => {
    if (document.querySelector('.tooltip-icon-legend')) {
        clearInterval(tooltipInterval);
        initializeSmartTooltips();
    }
}, 200);
</script>
"""
redirect_submission_on_close_script = """
<script>
function initializeRedirectObserver() {
    const successModal = document.querySelector('#success-modal');
    
    if (successModal) {
        const observer = new MutationObserver((mutationsList) => {
            for (const mutation of mutationsList) {
                // We only care about changes to the 'class' attribute.
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    
                    // Check if the 'hide' class has been ADDED to the class list.
                    // This is how Gradio hides the modal.
                    if (successModal.classList.contains('hide')) {
                        console.log("Success modal was closed. Redirecting to homepage...");
                        // This is the command to redirect the browser.
                        window.location.href = '/home';
                    }
                }
            }
        });

        // Tell the observer to watch the modal for attribute changes.
        observer.observe(successModal, { attributes: true });
    }
}

// Polling mechanism to wait for Gradio to build the UI.
const redirectInterval = setInterval(() => {
    if (document.querySelector('#success-modal')) {
        clearInterval(redirectInterval);
        initializeRedirectObserver();
    }
}, 200);
</script>
"""
# --- Theme Definition ---
theme = gr.themes.Base(
    primary_hue=gr.themes.Color(c100="#CFF5E8", c200="#B7EFDD", c300="#9FEAD1", c400="#87E5C5", c50="#E7FAF3", c500="#6FE0BA", c600="#57DBAF", c700="#3FD5A3", c800="#27D09C", c900="#0FCB8C", c950="#0fcb8c"),
    secondary_hue=gr.themes.Color(c100="#FCDCEB", c200="#FBCBE1", c300="#F9BAD7", c400="#F7A8CD", c50="#FDEEF5", c500="#F697C4", c600="#F586BA", c700="#F375B0", c800="#F263A6", c900="#F0529C", c950="#F0529C"),
    neutral_hue=gr.themes.Color(c100="#FDF9F4", c200="#C9C9C3", c300="#B0B5AF", c400="#97A09C", c50="#FAF2E9", c500="#7F8C89", c600="#667876", c700="#344F4F", c800="#1C3A3C", c900="#032629", c950="032629"),
    font=[gr.themes.GoogleFont('Manrope'), 'ui-sans-serif', 'sans-serif', 'sans-serif'],
    font_mono=[gr.themes.GoogleFont('Roboto Mono'), 'ui-monospace', 'monospace', 'monospace'],
).set(
    body_text_color='*neutral_950',
    body_text_color_subdued='*neutral_950',
    body_text_color_subdued_dark='*neutral_50',
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
    button_primary_text_color_dark='*neutral_900',
    block_border_color="#032629",
    block_border_color_dark="#9fead1",
    block_background_fill_dark="#032629",
    block_background_fill="#FAF2E9",
    checkbox_label_text_color="#032629",
    checkbox_label_background_fill="#D8D6CF",
    checkbox_label_background_fill_dark="#254243",
    checkbox_background_color_selected="#F0529C",
    checkbox_background_color_selected_dark="#0FCB8C",
)
try:
    with open(LOGO_PATH, "r") as f:
        svg_content = f.read()
    encoded_svg = urllib.parse.quote(svg_content)
    home_icon_data_uri = f"data:image/svg+xml,{encoded_svg}"
except FileNotFoundError:
    print(f"Warning: Home icon file not found at {LOGO_PATH}.")
    home_icon_data_uri = "none"

# --- This is the final CSS ---
final_css = css + f"""
/* --- Find the "Home" button and replace its text with an icon --- */
.nav-holder nav a[href$="/"] {{
    display: none !important;
}}
.nav-holder nav a[href*="/home"] {{
    grid-row: 1 !important;
    grid-column: 1 !important;
    justify-self: start !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    /* 2. Hide the original "Home" text */
    font-size: 0 !important;
    text-indent: -9999px;

    /* 3. Apply the icon as the background */
    background-image: url("{home_icon_data_uri}") !important;
    background-size: contain !important;
    background-repeat: no-repeat !important;
    background-position: center !important;

    width: 240px !important;    
    height: 50px !important;   
    padding: 0 !important;
    border: none !important;
    outline: none !important;
}}
"""
# --- Gradio App Definition ---
demo = gr.Blocks(
    theme=theme,
    css=final_css,
    head=scroll_script + redirect_script + tooltip_script + redirect_submission_on_close_script,
    title="AstaBench Leaderboards",
)
with demo.route("Home", "/home"):
    build_main_page()

with demo.route("Literature Understanding", "/literature-understanding"):
    build_lit_page()

with demo.route("Code & Execution", "/code-execution"):
    build_c_and_e_page()

with demo.route("Data Analysis", "/data-analysis"):
    build_data_analysis_page()

with demo.route("End-to-End Discovery", "/discovery"):
    build_e2e_page()

with demo.route("About", "/about"):
    build_about_page()

with demo.route("🚀 Submit an Agent", "/submit"):
    build_submission_page()
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
        demo.launch(debug=True, allowed_paths=["assets"], favicon_path="assets/favicon/favicon.ico")
    else:
        print("Launching in Space mode.")
        # For Spaces, share=False is typical unless specific tunneling is needed.
        # debug=True can be set to False for a "production" Space.
        demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=False, allowed_paths=["assets"], favicon_path="assets/favicon/favicon.ico")

