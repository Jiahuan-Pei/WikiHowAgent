import getpass
import os
from typing import List
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.adapters.openai import convert_message_to_dict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
# Set up LangGraph to orchestrate the simulated workflow
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Set API Key if not defined
def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

_set_if_undefined("OPENAI_API_KEY")

# Utility function to set up LLM and embeddings
from utils import setup_llm_and_embeddings  # Assuming this is defined in utils.py


# UserLLMAgent (User Simulation Agent)
class UserLLMAgent:
    def __init__(self, config_file='conf/ollma-llama3.yaml'):
        self.prompt_template = PromptTemplate(
            input_variables=["instruction", "context"],
            template=(
                "You are an active learner engaging with a system to deeply understand instructions. "
                "Given the learning step instruction and the current context, please respond to the system input. "
                "If the instruction is unclear or ambiguous, ask clarifying questions, otherwise say 'next step'. "
                "Be specific and ensure your questions are relevant to the instruction. "
                "Your responses should be brief, clear, concise, straight-out and coherent. Do not repeat."
                "When you are finished with the conversation, respond with a single word 'FINISHED'"
                "Context: {context}\n"
                "Teacher: {instruction}\n"
                "Learner:"
            ),
        )
        self.memory = ConversationBufferMemory(memory_key="context", return_messages=False)
        self.llm, _, self.params = setup_llm_and_embeddings(config_file)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            memory=self.memory,
            verbose=True
        )

    def run(self, instruction: str) -> str:
        return self.chain.run(instruction=instruction)

    def reset_memory(self):
        self.memory.clear()

    def add_context(self, context: str):
        self.memory.save_context({"instruction": ""}, {"context": context})


# TeacherLLMAgent (System Teaching Agent)
class TeacherLLMAgent:
    def __init__(self, config_file='conf/ollma-llama3.yaml', tutorial_document=None):
        self.tutorial_document = tutorial_document
        self.current_step = 0  # Starts at the first step
        self.total_steps = len(tutorial_document)  # Total number of steps
        self.prompt_template = PromptTemplate(
            input_variables=["current_step", "current_step_instruction", "context", "user_utterance"],
            template=(
                "You are a teacher interacting with a learner. "
                "Your task is to provide instructions or answer questions based on the current step "
                "from the tutorial document. If the learner asks a question, clarify it. If they indicate readiness, "
                "move to the next step. If this is the final step, indicate that the tutorial is complete."
                "Step {current_step} instruction: {current_step_instruction}\n"
                "Context: {context}\n"
                "Learner: {user_utterance}\n"
                "Teacher:"
            ),
        )
        self.memory = ConversationBufferMemory(memory_key="context", input_key="current_step_instruction", return_messages=False)
        self.llm, _, self.params = setup_llm_and_embeddings(config_file)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            memory=self.memory,
            verbose=True
        )

    def run(self, user_utterance) -> str:
        # Prevent overshooting the last step
        if self.current_step >= self.total_steps:
            return "FINISHED"  # End the tutorial

        # Prepare current step tutorial content
        current_step_instruction = self.tutorial_document[self.current_step]

        # Only the first teacher utterance summerize what to learn today.
        if self.current_step == 0:
            system_response=self.llm.invoke(
            "You are a helpful teacher assistant. Summerize what you will teach today."
            f"Introduction: {'\n'.join(tutorial_document)}"
            ).content
            # self.current_step += 1

        # Run the LLM chaåin
        system_response = self.chain.run(
            current_step=str(self.current_step + 1),
            current_step_instruction=current_step_instruction,
            user_utterance=user_utterance
        )

        # Check if user wants to proceed to the next step
        next_step = self.llm.invoke(
            "Based on the user's response, does the user understand the current step "
            "and want to proceed to the next step? Respond with 'yes' or 'no' only. "
            f"User's response: {user_utterance}"
        ).content
        # if any(phrase in user_utterance.lower() for phrase in ["understand", "next step"]):
        if next_step == 'yes' and self.current_step>0:
            self.current_step += 1

        # Check if we've reached the final step
        if self.current_step >= self.total_steps:
            return "Step FINAL\t" + system_response + "\nThe tutorial is now complete. FINISHED."

        return f"**Step {self.current_step + 1}**" + system_response


def my_chat_bot(messages):
    """
    Generate a response from the chat bot using the TeacherLLMAgent.

    :param messages: A list of message dictionaries containing the conversation history.
    :return: The response from the chat bot as a dictionary.
    """
    # Initialize the TeacherLLMAgent (assuming it requires a tutorial document)
    teacher_agent = TeacherLLMAgent(tutorial_document=tutorial_document)

    # Extract the latest user message from the conversation history
    user_message = messages[-1] if messages else None

    if user_message and user_message['role'] == 'user':
        # Get the instruction from the user's message
        user_utterance = user_message['content']

        # Use the TeacherLLMAgent to get the instruction for the current step
        response = teacher_agent.run(user_utterance)

        # Create a response message
        chat_bot_response = {
            "role": "assistant",
            "content": response
        }
    else:
        # Use the TeacherLLMAgent to get the instruction for the current step
        user_utterance = '\n'.join(tutorial_document)
        response = teacher_agent.run(user_utterance)
        # Handle the case where there is no valid user message
        chat_bot_response = {
            "role": "assistant",
            "content": f"I'm grad to assist you with learning! {response}"
        }

    return chat_bot_response


def chat_bot_node(state):
    messages = state["messages"]
    messages = [convert_message_to_dict(m) for m in messages]
    chat_bot_response = my_chat_bot(messages)
    return {"messages": [AIMessage(content=chat_bot_response["content"])]}


def _swap_roles(messages):
    """
    Swap roles between human and AI messages for the user simulation.
    """
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            new_messages.append(HumanMessage(content=m.content))
        else:
            new_messages.append(AIMessage(content=m.content))
    return new_messages


def simulated_user_node(state):
    """
    Simulate a user node in the workflow.
    """
    messages = state["messages"]
    
    # Swap roles to prepare for the UserLLMAgent
    new_messages = _swap_roles(messages)
    
    # Initialize the UserLLMAgent
    user_agent = UserLLMAgent()

    # Use the UserLLMAgent to simulate a user response
    response = user_agent.run(new_messages[-1].content)  # Access the 'content' attribute directly

    # Create a new message for the simulated user's response
    simulated_user_message = {
        "role": "user",
        "content": response
    }

    # Append the simulated user's response to the messages
    messages.append(simulated_user_message)

    return {"messages": messages}

def should_continue(state, max_step=25): 
    """
    Determine whether the simulation should continue or end based on message content and step count.
    """
    messages = state["messages"]
    if len(messages) > max_step:
        return "end"
    elif any("FINISHED" in m.content for m in messages):
        return "end"
    else:
        return "continue"



# Initialize the tutorial document and agents
howto_query = "Make Colored Salt"
tutorial_document = [
    "Fill a container with salt. A jug or pitcher, a deep bowl, a plastic food container, etc. will all suffice.",
    "Squeeze a little tempera paint into the salt.",
    "Mix with a spoon or other item. Stir until the paint is evenly distributed through the salt.",
    "Let stand overnight to dry. Make as many more colors as you need for your project. That way, they'll all be ready at the same time.",
    "Test before using. Check that the salt has dried before using in your craft, rangoli, teaching, etc. projects."
]
max_step = len(tutorial_document)
teacher_agent = TeacherLLMAgent(tutorial_document=tutorial_document)
user_agent = UserLLMAgent()

# Building the LangGraph for the flow
graph_builder = StateGraph(State)
graph_builder.add_node("user", simulated_user_node)
graph_builder.add_node("chat_bot", chat_bot_node)
graph_builder.add_edge("chat_bot", "user")
graph_builder.add_conditional_edges(
    "user", 
    should_continue, 
    {"end": END, "continue": "chat_bot"}
)
graph_builder.add_edge(START, "chat_bot")
simulation = graph_builder.compile()

conversations = []
# Running the simulation
for chunk in simulation.stream({"messages": []}):
    if END not in chunk:
        # Assuming chunk contains messages, append them to all_messages
        # print(chunk)  # Print the chunk for debugging or logging
        print(chunk)
        print("----")
        if 'user' in chunk:
            user_uttr = f">>>>>>>>>Learner: {chunk['user']['messages'][-1]['content'].strip()}"
            conversations.append(user_uttr)
        elif 'chat_bot' in chunk:
            sys_uttr = f'>>>>>>>>>Teacher: {chunk['chat_bot']['messages'][-1].content}'
            conversations.append(sys_uttr)
print("="*50)
for i, uttr in enumerate(conversations):
    print(i, uttr)
    print("----")
        


