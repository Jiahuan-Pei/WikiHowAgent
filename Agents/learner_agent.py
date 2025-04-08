from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.util import setup_llm_and_embeddings


# Learner Agent
class LearnerLLMAgent:
    def __init__(self, config):
        super().__init__()
        self.llm, _, self.config = setup_llm_and_embeddings(config)

        self.prompt_template = PromptTemplate(
            input_variables=["instruction"],
            template=(
                "You are a 🤔 student learning from a teacher via multi-turn conversations.\n\n"
                "🎯 Your role:\n"
                "- Read and understand the teacher’s instructions.\n"
                "- Respond naturally and concisely.\n"
                "- If the step is **clear**, acknowledge it briefly and ask to move to the next step by appending the token 'NEXT'.\n"
                "- If the step is **unclear**, ask a **brief and specific** question.\n"
                "- If the teacher has just started by mentioning the token 'BEGIN', do not ask questions.\n"
                "- If the teacher says **'FINISHED'** or acknowledges completion, **thank them briefly** without asking further questions.\n\n"
                "Teacher: {instruction}\n"
                "Learner:"
            ),
        )
        
        self.chain = LLMChain(prompt=self.prompt_template, llm=self.llm, verbose=True)
        # self.chain =  RunnableSequence(self.prompt_template, self.llm)

    def respond(self, data: dict) -> str:
        try:
            learner_response = self.chain.invoke({"instruction": data["instruction"]})['text']
        except:
            print('Learner:INVOKE:ERR', data)

        print('<<🤔 Learner>>', learner_response)
        return learner_response
    
    def ask(self, data: dict) -> str:
        return
    
    def feedback(self, data: dict) -> str:
        return    