import gradio as gr


with gr.Blocks() as demo:
    gr.Markdown("# Code & Execution")
    gr.Markdown("This is some text to explain what Code & Execution metrics are!")
    t = gr.Textbox()
    demo.load(lambda : "Yipeeee", None, t)

if __name__ == "__main__":
    demo.launch()