from io import BytesIO
from pypdf import PdfReader
from langchain.schema import Document

def load_document(file, encoding='iso-8859-13'):
    documents = []
    
    with open(file, 'rb') as file_obj:
        pdf_reader = PdfReader(BytesIO(file_obj.read()))
        
        def decode_lithuanian(text):
            try:
                return text.decode(encoding, 'replace')
            except:
                return text.decode('utf-8', 'replace')
        
        pdf_reader.stream.decode_text = decode_lithuanian
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():
                metadata = {"source": file, "page": page_num + 1}
                documents.append(Document(page_content=text, metadata=metadata))
    
    return documents