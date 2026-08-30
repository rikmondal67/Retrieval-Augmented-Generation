from llama_parse import LlamaParse
from app.core.config import LLAMA_CLOUD_API_KEY

parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    result_type="markdown",   # ask LlamaParse to return clean markdown
    verbose=True,
)