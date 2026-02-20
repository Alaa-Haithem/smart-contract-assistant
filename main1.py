import gradio as gr
from ingestion import ingest_document
from retrieval import create_qa_chain

qa_chain = None

def upload_file(file):
    global qa_chain
    ingest_document(file.name)
    qa_chain = create_qa_chain()
    return "Document uploaded and processed successfully!"

def chat(question):
    if qa_chain is None:
        return "Please upload a document first."
    result = qa_chain(question)
    return result["result"]

with gr.Blocks() as demo:
    gr.Markdown("# Smart Contract Summary & Q&A Assistant")

    with gr.Tab("Upload Document"):
        file = gr.File()
        upload_btn = gr.Button("Upload")
        upload_output = gr.Textbox()
        upload_btn.click(upload_file, file, upload_output)

    with gr.Tab("Chat"):
        question = gr.Textbox(label="Ask a question")
        answer = gr.Textbox(label="Answer")
        question.submit(chat, question, answer)

demo.launch()
