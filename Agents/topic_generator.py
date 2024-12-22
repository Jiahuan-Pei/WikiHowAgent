from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

def generate_queries_with_langchain(capabilities):
    """Generate a list of queries using LangChain."""
    prompt_template = """
    You are a creative thinker. Generate an ultimate complex How-to task by to thoroughly understand the subtasks you are able to do by a list of specific, actionable questions {capabilities}.
    """
    prompt = PromptTemplate(input_variables=["topic"], template=prompt_template)
    formatted_prompt = prompt.format(capabilities=capabilities)
    response = llm(formatted_prompt)
    return response.split("\n")