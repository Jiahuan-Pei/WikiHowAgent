from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.util import setup_llm_and_embeddings
from concurrent.futures import ThreadPoolExecutor

# Teacher Agent
class TeacherLLMAgent:
    def __init__(self, config, tutorial_document=None):
        self.tutorial_document = tutorial_document
        self.current_step = 1
        self.total_steps = len(tutorial_document) - 1  # 0th index is the summary
        self.llm, _, self.config = setup_llm_and_embeddings(config)
        self.dialogue_history = []  # Stores the history of conversation

        self.prompt_template = PromptTemplate(
            input_variables=["history", "current_step", "instruction", "user_utterance"],
            template=(
                "You are a teacher guiding a learner through a tutorial step by step via multi-turn conversations. "
                "Your job is to give instructions for the current step and answer any questions."
                "Conversation history:\n{history}\n\n"
                "Summary of tutorial: {summary}\n"
                "Instruction (Step {current_step}): {instruction}\n"
                "Learner: {user_utterance}\n"
                "Teacher:"
            ),
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=True)
        # self.chain = self.prompt_template | self.llm
        # self.chain =  RunnableSequence(self.prompt_template, self.llm)

    def run(self, user_utterance: str) -> str:
        """Process the learner's response and generate the teacher's reply."""
        if self.current_step > self.total_steps:
            return "FINISHED"

        summary = self.tutorial_document[0]  # Summary of the tutorial
        instruction = self.tutorial_document[self.current_step]

        # Create history string
        history = "\n".join(self.dialogue_history[-5:])  # Keep last 5 exchanges for brevity


        # Generate teacher response
        teacher_response = self.chain.run(
            history=history,
            summary=summary, 
            current_step=self.current_step,
            instruction=instruction,
            user_utterance=user_utterance
        )

        # Update memory
        self.dialogue_history.append(f"Learner: {user_utterance}")
        self.dialogue_history.append(f"Teacher: {teacher_response}")       

        # Check if the learner asked a question
        is_question = "?" in user_utterance
        if is_question:
            return teacher_response  # Answer question without moving forward

        # Determine if the learner wants to proceed
        next_step_decision = self.llm.invoke(
            f"Based on this response: '{user_utterance}', does the learner understand and want to proceed? "
            f"Reply with 'yes' or 'no'."
        ).content.lower()

        if "yes" in next_step_decision or self.current_step < self.total_steps:
            self.current_step += 1  # Move to the next step

        if self.current_step > self.total_steps:
            return f"Step FINAL: {teacher_response}\nThe tutorial is now complete. FINISHED."

        return teacher_response
    
    def run_batch(self, user_utterances: list, batch_size=2) -> list:
        """Process multiple learner responses in parallel."""
        texts = [{
            "history": "\n".join(self.dialogue_history[-5:]),  # Include history
            "summary": self.tutorial_document[0],
            "current_step": self.current_step,
            "instruction": self.tutorial_document[self.current_step],
            "user_utterance": utterance
        } for utterance in user_utterances]

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(executor.map(self.chain.run, texts))

        # Update dialogue memory for all batch responses
        for user_utterance, response in zip(user_utterances, results):
            self.dialogue_history.append(f"Learner: {user_utterance}")
            self.dialogue_history.append(f"Teacher: {response}")            
        return results
