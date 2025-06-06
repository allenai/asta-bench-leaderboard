import gradio as gr


with gr.Blocks() as demo:
    gr.Markdown("# Literature Understanding")
    gr.Markdown("This is some text to explain what literature Understanding metrics are!")
    t = gr.Textbox()
    demo.load(lambda : "Loaded", None, t)

if __name__ == "__main__":
    demo.launch()