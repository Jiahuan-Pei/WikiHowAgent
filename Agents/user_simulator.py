from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from Agents.util import setup_llm_and_embeddings 

class UserLLMAgent:
    """
    A class representing a user simulator that acts as an active learner.
    The agent engages with a system to ask clarifying and exploratory questions
    based on instructions.
    """
    
    def __init__(self, config_file='conf/ollma-llama3.yaml'):
        """
        Initialize the UserLLMAgent with a language model, memory, and behavior prompt.
        
        :param model_name: The name of the language model to use (default: "gpt-4").
        :param temperature: The temperature setting for the LLM's creativity (default: 0.7).
        """
        self.prompt_template = PromptTemplate(
            input_variables=["instruction", "context"],
            template=(
                "You are an active learner engaging with a system to understand instructions deeply. "
                "You are curious, critical, and focused on learning. Given the following learning step: "
                "'{instruction}', and the current context: '{context}', ask a thoughtful question. "
                "Your questions should aim to clarify, explore, or confirm understanding. "
                "Be specific and ensure your questions are relevant to the instruction."
            ),
        )
        self.memory = ConversationBufferMemory(memory_key="context", input_key="instruction")
        self.llm, _, self.params = setup_llm_and_embeddings(config_file)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            memory=self.memory,
            verbose=True
        )

    def simulate_step(self, instruction: str) -> str:
        """
        Simulate the agent engaging with a single learning step.
        
        :param instruction: The instruction or step provided by the system.
        :return: The agent's generated questions as a response.
        """
        return self.chain.run(instruction=instruction)

    def reset_memory(self):
        """Clear the agent's memory to start fresh."""
        self.memory.clear()

    def add_context(self, context: str):
        """
        Add context to the agent's memory.
        
        :param context: Additional information to add to the current memory.
        """
        self.memory.save_context({"instruction": ""}, {"context": context})

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    user_agent = UserLLMAgent()
    
    # Simulate a learning step
    instruction = "To complete this step, connect the red wire to the terminal marked 'A'."
    response = user_agent.simulate_step(instruction)
    print("User Agent's Response:\n", response)
    
    # Add additional context and simulate another step
    user_agent.add_context("Ensure the power supply is off before working with the wires.")
    instruction2 = "Now, attach the green wire to the terminal marked 'B'."
    response2 = user_agent.simulate_step(instruction2)
    print("User Agent's Response:\n", response2)
    
    # Reset memory
    user_agent.reset_memory()
