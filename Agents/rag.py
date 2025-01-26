import numpy as np
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Utility function to set up LLM and embeddings
from utils import setup_llm_and_embeddings

class RAG:
    def __init__(self, config_file='conf/ollma-llama3.yaml'): # gpt-4o
        self.llm, self.embeddings, self.param = setup_llm_and_embeddings(config_file)
        self.doc_embeddings = None
        self.docs = None

    def load_documents(self, documents):
        """Load documents and compute their embeddings."""
        self.docs = documents
        self.doc_embeddings = self.embeddings.embed_documents(documents)

    def get_most_relevant_docs(self, query):
        """Find the most relevant document for a given query."""
        if not self.docs or not self.doc_embeddings:
            raise ValueError("Documents and their embeddings are not loaded.")

        query_embedding = self.embeddings.embed_query(query)
        
        similarities = []
        for doc_emb in self.doc_embeddings:
            if np.linalg.norm(query_embedding) == 0 or np.linalg.norm(doc_emb) == 0:
                similarities.append(0)  # Handle zero norm case
            else:
                similarity = np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
                similarities.append(similarity)

        most_relevant_doc_index = np.argmax(similarities)
        return [self.docs[most_relevant_doc_index]]

    def generate_answer(self, query, relevant_doc):
        """Generate an answer for a given query based on the most relevant document."""
        prompt = f"question: {query}\n\nDocuments: {relevant_doc}"
        messages = [
            ("system", "You are a helpful assistant that answers questions based on given documents only."),
            ("human", prompt),
        ]
        ai_msg = self.llm.invoke(messages)
        return ai_msg.content

def demo():
    sample_docs = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
    ]
    sample_queries = [
        "Who introduced the theory of relativity?",
        "Who was the first computer programmer?",
        "What did Isaac Newton contribute to science?",
        "Who won two Nobel Prizes for research on radioactivity?",
        "What is the theory of evolution by natural selection?"
    ]

    expected_responses = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
    ]

    # Initialize RAG instance
    rag = RAG()

    # Load documents
    rag.load_documents(sample_docs)

    # Query and retrieve the most relevant document
    query = "Who introduced the theory of relativity?"
    relevant_doc = rag.get_most_relevant_docs(query)

    # Generate an answer
    answer = rag.generate_answer(query, relevant_doc)

    print(f"Query: {query}")
    print(f"Relevant Document: {relevant_doc}")
    print(f"Answer: {answer}")

    dataset = []

    for query,reference in zip(sample_queries,expected_responses):

        relevant_docs = rag.get_most_relevant_docs(query)
        response = rag.generate_answer(query, relevant_docs)
        dataset.append(
            {
                "user_input":query,
                "retrieved_contexts":relevant_docs,
                "response":response,
                "reference":reference
            }
        )
    from ragas import EvaluationDataset
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    print("Evaluation Dataset:", evaluation_dataset)

    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper

    # llm = ChatOllama(model="llama3",verbose=False,timeout=600,num_ctx=8192,disable_streaming=False)
    llm = ChatOllama(model="llama3",timeout=1600, temperature=0)
    evaluator_llm = LangchainLLMWrapper(llm)

    from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

    # Before evaluation
    print("Evaluating with dataset:", evaluation_dataset)


    # Perform evaluation
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm
    )

    # Check the result
    print("Evaluation Result:", result)

    # import os
    # os.environ["RAGAS_APP_TOKEN"] = "your_app_token"
    # result.upload()
