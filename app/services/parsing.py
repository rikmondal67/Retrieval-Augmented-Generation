from app.clients.llamaparse_client import parser
import requests

def extract_markdown(file_url: str) -> str:


    response = requests.get(file_url)
    response.raise_for_status() 
    
  
    file_bytes = response.content
    
    documents = parser.load_data(
        file_bytes, 
        extra_info={"file_name": "document.pdf"} 
    )
    
    res = "\n\n".join(doc.text for doc in documents)
    # print(res)
    return res