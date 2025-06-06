import gradio as gr


with gr.Blocks() as demo:
    gr.Markdown("# Data Analysis")
    gr.Markdown("This is some text to explain what Data Analysis metrics are!")
    t = gr.Textbox()
    demo.load(lambda : "It's another page man", None, t)

if __name__ == "__main__":
    demo.launch()