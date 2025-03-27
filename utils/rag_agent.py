import os
import time
import faiss
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from utils.util import setup_llm_and_embeddings
import joblib


def load_all_documents(root_dir, suffix=".md", limit=None):
    """
    Load all markdown documents from the specified directory.
    
    Args:
        root_dir (str): Path to the root directory containing markdown files.
        suffix (str): File suffix to filter (default: '.md').
        limit (int): Limit the number of documents loaded (default: None).

    Returns:
        list: Loaded documents as LangChain document objects.
    """
    documents = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(suffix):
                file_path = os.path.join(subdir, file)
                try:
                    loader = TextLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return documents if not limit else documents[:limit]


class RAGAgent:
    """
    Retrieval-Augmented Generation (RAG) Agent for answering questions using LLMs and document retrieval.
    """
    def __init__(self, config_file="conf/ollma-llama3.yaml"):
        # Setup LLM and embeddings
        self.llm, self.embeddings, self.params = setup_llm_and_embeddings(config_file)
        self.verbose = self.params.get("verbose", True)
        self.vector_store_file = self.params.get("index_file", "./faiss_index")
        self.k = self.params.get("k", 5)

        # Document and retrieval setup
        self.documents = []
        self.vector_store = None
        self.retriever = None

        # Load documents and initialize retriever
        self._initialize_retriever(self.params["root_doc_dir"])

        # Setup prompt and LLMChain
        self.prompt = PromptTemplate(
            input_variables=["input", "retrieved_docs"],
            template="""You are a helpful assistant. Please provide a brief instruction or answer to the following question using the retrieved documents. If no relevant documents are found, inform the user that you cannot assist and conclude with "END".

Retrieved Documents:
{retrieved_docs}

Question:
{input}
"""
        )
        # self.chain = LLMChain(prompt=self.prompt, llm=self.llm)
        self.chain = self.prompt | self.llm

    def _initialize_retriever(self, root_dir):
        """
        Initialize retriever by loading existing FAISS index or creating a new one.
        """
        if os.path.exists(self.vector_store_file):
            self._load_existing_index()
        else:
            self._build_new_index(root_dir)

    def _load_existing_index(self):
        """Load retriever from saved state."""
        try:
            print(f"Loading existing retriever {self.vector_store_file}...")
            # Load the vector_store from the file
            self.vector_store = FAISS.load_local(self.vector_store_file, self.embeddings, allow_dangerous_deserialization=True)
            self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={'k': self.k})
            print("Retriever loaded successfully.")
        except Exception as e:
            print(f"Error loading retriever: {e}")
            print("Falling back to building a new index.")

    def _build_new_index(self, root_dir):
        """Build a new retriever from documents."""
        start_time = time.time()
        self.documents = load_all_documents(root_dir)
        if self.verbose:
            print(f"Loaded {len(self.documents)} documents to build a new retriever {self.vector_store_file}.")

        try:
            # Create the vector store from documents
            self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
            
            # Save the vector_store index to a file
            self.vector_store.save_local(self.vector_store_file)
            self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={'k': self.k})
            
            if self.retriever:  # Ensure retriever is not None
                print(f"New vector store created and saved to {self.vector_store_file}.")
            else:
                print("Retriever was not created successfully.")
        except Exception as e:
            print(f"Error building retriever: {e}")
        finally:
            print(f"Indexing completed in {time.time() - start_time:.2f} seconds.")

    def retrieve_documents(self, query):
        """
        Retrieve relevant documents based on the query.

        Args:
            query (str): Query string.

        Returns:
            str: Combined content of retrieved documents.
        """
        try:
            docs = self.retriever.get_relevant_documents(query)
            if self.verbose:
                print(f"\n=====>Retrieved {len(docs)} docs:\n", retrieved_docs)
            return "\n".join(doc.page_content for doc in docs)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return ""

    def run(self, input_text):
        """
        Generate a response to the input text using retrieved documents.

        Args:
            input_text (str): User query.

        Returns:
            str: Response from the LLM.
        """
        try:
            retrieved_docs = self.retrieve_documents(input_text)
            if not retrieved_docs:  # Check if no documents were retrieved
                return "Sorry, I cannot assist with your question as I have not found a relevant tutorial. END"  # Inform the user and conclude

            # Ensure input is a dictionary with the correct keys
            response = self.chain.invoke({"input": input_text, "retrieved_docs": retrieved_docs})
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while generating a response."


# Example usage
if __name__ == "__main__":
    rag_agent = RAGAgent()
    question = "How to Make Baskets"
    answer = rag_agent.run(question)
    print("\n=====>Final Answer:\n", answer)
