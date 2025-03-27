from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
# from langchain.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory #, ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.output_parsers import StrOutputParser
import traceback

# Define custom prompt template
class CustomPromptTemplate(PromptTemplate):
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\n"
        
        kwargs["agent_scratchpad"] = thoughts
        return self.template.format(**kwargs)

class LLMAgent:
    def __init__(self, model, model_type="ollama", temperature=0):
        # Initialize the LLM based on the specified model type
        if model_type == "openai":
            # export OPENAI_API_KEY="your-api-key" and it will be read from env var
            self.llm = ChatOpenAI(model=model, temperature=temperature) # "gpt-3.5-turbo"
        elif model_type == "ollama":
            self.llm = ChatOllama(model=model, temperature=temperature) # "llama3"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")        

        # Define tools (e.g., web search)
        search_tool = DuckDuckGoSearchRun()
        self.tools = [
            Tool(
                name="WebSearch",
                func=search_tool.run,
                description="Useful for answering questions by searching the web."
            )
        ]

        # Define custom prompt template
        self.prompt = CustomPromptTemplate(
            input_variables=["input", "chat_history", "intermediate_steps"],
            template="""You are a helpful assistant. Use the tools available if needed.

Tools: {tools}

Use the following format:
Thought: What to do next
Action: Tool name
Action Input: Input to the tool
Observation: Tool output
...(repeat until done)
Final Answer: Provide your response

Question: {input}
{agent_scratchpad}"""
        )

        # Initialize message history and memory
        self.chat_message_history = ChatMessageHistory()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=self.chat_message_history,
            return_messages=True
        )

        # Create agent with error handling
        self.agent = self._initialize_agent()

    def _initialize_agent(self):
        try:
            return initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent="chat-conversational-react-description",
                verbose=True,
                memory=self.memory,
                agent_kwargs={"prompt": self.prompt},
                handle_parsing_errors=True
            )
        except Exception as e:
            print(f"Error initializing agent: {str(e)}\n{traceback.format_exc()}")
            return None

    def run(self, input_text: str) -> str:
        """
        Run the agent with the given input text
        """
        if self.agent is None:
            return "Agent initialization failed."
        
        try:
            response = self.agent.run(input_text)
            return response
        except Exception as e:
            return f"Error occurred while running the agent: {str(e)}\n{traceback.format_exc()}"

# Example usage:
if __name__ == "__main__":
    # Create an instance of the LLMAgent
    # agent = LLMAgent(model_type="openai")  # Change to "ollama" to use Ollama
    agent = LLMAgent(model="llama3")  # Change to "ollama" to use Ollama

    # Test the agent
    # response = agent.run("What's the latest news about AI research?")
    # print(response)
    response = agent.run("How to train a dog?")
    print(response)