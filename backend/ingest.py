import sys
import os
import re
from langchain.schema import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from tools.document_loader import load_document
from backend.vectorstore import create_vector_store
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken


def chunk_text(text: str, metadata: dict = None, min_chunk_size: int = 500, max_chunk_size: int = 1500) -> list[Document]:
    """
    Split text content into chunks based on paragraphs and size constraints.
    
    :param text: The input text to be chunked.
    :param metadata: Optional metadata to be included with each chunk.
    :param min_chunk_size: Minimum size of a chunk in characters.
    :param max_chunk_size: Maximum size of a chunk in characters.
    :return: List of Document objects containing the chunks.
    """
    chunks = []
    paragraphs = re.split(r'\n\s*\n', text)
    
    current_chunk = []
    current_chunk_size = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        
        if not paragraph:
            continue
        
        if current_chunk_size + len(paragraph) > max_chunk_size and current_chunk_size >= min_chunk_size:
            chunks.append(Document(page_content='\n\n'.join(current_chunk), metadata=metadata))
            current_chunk = []
            current_chunk_size = 0
        
        current_chunk.append(paragraph)
        current_chunk_size += len(paragraph)
        
        if current_chunk_size >= min_chunk_size:
            if re.match(r'^[A-Z].*:$', paragraph) or paragraph.startswith('•') or re.match(r'^\d+\.', paragraph):
                chunks.append(Document(page_content='\n\n'.join(current_chunk), metadata=metadata))
                current_chunk = []
                current_chunk_size = 0

    if current_chunk:
        chunks.append(Document(page_content='\n\n'.join(current_chunk), metadata=metadata))

    return chunks


def chunk_table_data(
    table_data: list[tuple[list[str], list[list[str]]]],
    max_rows: int = 20,
    overlap: int = 2,
) -> list[str]:
    """Split table data into chunks, including headers in each chunk."""
    table_chunks = []
    for header, rows in table_data:
        for i in range(0, len(rows), max_rows - overlap):
            chunk = [header] + rows[i : i + max_rows]
            table_text = "\n".join([",".join(row).strip() for row in chunk])
            table_chunks.append(table_text.strip())
    return table_chunks



def calculate_embedding_cost(chunks):
    """Calculates the embedding cost for a list of chunks using the OpenAI Ada 002 tokenizer."""
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum(len(encoding.encode(chunk.page_content)) for chunk in chunks)
    cost = (total_tokens / 1000) * 0.0004
    return cost


def main():
    index_name = 'diet-plan-chatbot'
    pdf_file = r'C:\\Users\\tadas\Desktop\\rag\diet-plan-chatbot\\mitybos_planas_2.pdf'

    documents = load_document(pdf_file)
    
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc.page_content, doc.metadata)
        all_chunks.extend(chunks)


    create_vector_store(index_name, all_chunks)


if __name__ == '__main__':
    main()
