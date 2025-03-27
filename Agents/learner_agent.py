from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.util import setup_llm_and_embeddings
from concurrent.futures import ThreadPoolExecutor
# Learner Agent
class LearnerLLMAgent:
    def __init__(self, config):
        self.llm, _, self.config = setup_llm_and_embeddings(config)
        self.dialogue_history = []  # Stores past dialogue exchanges

        self.prompt_template = PromptTemplate(
            input_variables=["instruction"],
            template=(
                "You are a student learning from a teacher via multi-turn conversations. "
                "Read the teacher's instruction and respond naturally. "
                "If the step is clear, acknowledge it shortly in a conversational way. "
                "If unclear, ask a brief and specific question about what is confusing."
                "If the teacher mentions 'FINISHED' or acknowledges the completion of the tutorial, respond by briefly thanking the teacher. Do not ask questions.\n\n"
                "Conversation history:\n{history}\n\n"
                "Teacher: {instruction}\n"
                "Learner:"
            ),
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=True)

    def run(self, instruction: str) -> str:
        # Create history string (limit to last 5 exchanges for brevity)
        history = "\n".join(self.dialogue_history[-5:])

        # Generate response
        learner_response = self.chain.run(history=history, instruction=instruction)

        # Update memory
        self.dialogue_history.append(f"Teacher: {instruction}")
        self.dialogue_history.append(f"Learner: {learner_response}")

        return learner_response
    
    def run_batch(self, instructions: list, batch_size=2) -> list:
        """Run multiple learner responses in parallel."""
        texts = [{
            "history": "\n".join(self.dialogue_history[-5:]),  # Include recent history
            "instruction": t
        } for t in instructions]

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(executor.map(self.chain.run, texts))

        # Update dialogue memory for all batch responses
        for instr, response in zip(instructions, results):
            self.dialogue_history.append(f"Teacher: {instr}")
            self.dialogue_history.append(f"Learner: {response}")
        return results  