from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

def generate_queries_with_langchain(topic):
    """Generate a list of queries using LangChain."""
    prompt_template = """
    You are a helpful assistant. Generate a list of specific, actionable questions
    to thoroughly understand the topic: {topic}.
    """
    prompt = PromptTemplate(input_variables=["topic"], template=prompt_template)
    formatted_prompt = prompt.format(topic=topic)
    response = llm(formatted_prompt)
    return response.split("\n")
