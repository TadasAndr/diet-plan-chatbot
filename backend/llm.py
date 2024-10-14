from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class LLM:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o",
            streaming=True
        )
        self.setup_chain()

    def setup_chain(self):
        prompt_template = """You are an expert dietitian capable of creating personalized diet plans according to client needs.
        Carefully analyze the provided information and restrictions.

            Instructions:
            1. Always respond in Lithuanian.
            2. Provide detailed and specific dietary advice based on the given context and question.
            3. Consider any mentioned health conditions, allergies, or dietary restrictions.
            4. If the question is unclear or lacks necessary information, ask for clarification.
            5. Include balanced meal suggestions and portion recommendations when appropriate.
            6. Offer alternatives for common food intolerances or allergies.
            7. If the question is outside your expertise or not related to diet, politely redirect the conversation back to dietary topics.

            Context: {context}

            Question: {question}

            Answer (in Lithuanian):"""

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )


    def ask(self, question, session_id, stream_handler=None):
        callbacks = [stream_handler] if stream_handler else None
            
        # Debug: Print retrieved documents
        retrieved_docs = self.vector_store.as_retriever().get_relevant_documents(question)
        print("Retrieved documents:")
        for i, doc in enumerate(retrieved_docs):
            print(f"Document {i + 1}:")
            print(f"Content: {doc.page_content[:200]}...")  # Print first 200 characters
            print(f"Metadata: {doc.metadata}")
            print("---")

        response = self.chain({"query": question}, callbacks=callbacks)
        return response['result']