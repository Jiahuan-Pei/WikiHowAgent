from rag import RAG
from langchain_ollama import ChatOllama
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

rag = RAG()
# Load documents
rag.load_documents(sample_docs)

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
from ragas.metrics import AspectCritic

# Before evaluation
print("Evaluating with dataset:", evaluation_dataset)

metric = AspectCritic(name="summary_accuracy",llm=evaluator_llm, definition="Verify if the summary is accurate.")


from ragas.metrics import FactualCorrectness
scorer = FactualCorrectness(llm=evaluator_llm)

# Perform evaluation
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), metric],
    llm=evaluator_llm
)

# Check the result
print("Evaluation Result:", result)


# import os
# os.environ["RAGAS_APP_TOKEN"] = "your_app_token"
# result.upload()
