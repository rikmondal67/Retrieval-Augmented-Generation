from app.clients.llamaparse_client import parser

def extract_markdown(pdf_path: str) -> str:
    """
    Sends the PDF to LlamaCloud's LlamaParse API and returns the
    parsed content as a single markdown string.
    """
    documents = parser.load_data(pdf_path)
    return "\n\n".join(doc.text for doc in documents)