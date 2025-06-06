import gradio as gr


with gr.Blocks() as demo:
    gr.Markdown("# End-to-End Evaluation")
    gr.Markdown("This is some text to explain what E2E metrics are!")
    t = gr.Textbox()
    demo.load(lambda : "It's another page woman", None, t)

if __name__ == "__main__":
    demo.launch()