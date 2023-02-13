import gradio as gr
from gpt_index import GPTSimpleVectorIndex

index = GPTSimpleVectorIndex.load_from_disk('index_project.json')

def selected_file(files :str) -> str:
    #global index
    #if files == "Skryllegården":
    #    index = GPTSimpleVectorIndex.load_from_disk('index_project.json')
    #else:
    #    index = GPTSimpleVectorIndex.load_from_disk("index_books.json")

    return f"Selected {files}"

with gr.Blocks(title="AFRY X | Document Assistant") as demo:
    gr.Markdown("""# AFRY X | Document Assistant""")

    with gr.Row():
        gr.Markdown("Welcome to the afry documentation assistant! 🤖  \nUpload files if you wish to search custom documents. 👉")
        gr.UploadButton("Upload a file (default: Skryllegården)")
        selected = gr.Dropdown(choices=["Skryllegården", "Böcker"], value="Skryllegården", label="Select a file collection")
        out = gr.Markdown("Selected Skryllegården")
        selected.change(selected_file, inputs=selected, outputs=out)

    gr.Markdown("### Ask questions and you'll get an answer")
    chatbot = gr.Chatbot()
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

    def query(txt, history=[]):
        print(history)
        print(txt)
        """
        https://gpt-index.readthedocs.io/en/latest/guides/usage_pattern.html#setting-response-mode
        """
        merged_hist = " ".join(a+b for a,b in history)
        resp = index.query(txt)
        print(resp.get_formatted_sources())
        conversation = [(txt, resp.response)]
        all_conversation = (history + conversation)

        return all_conversation, all_conversation


    txt.submit(query, [txt, state], [chatbot, state])

# index.docstore.docs.get("d657aac0-8075-4873-88a0-153fd402b93f") to retrieve source...!!!
# TODO:
#   - Add "reset"
#   - ...

demo.launch()
