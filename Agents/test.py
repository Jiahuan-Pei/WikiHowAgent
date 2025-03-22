# from langchain_huggingface.llms import HuggingFacePipeline
# import torch

# from langchain_core.prompts import PromptTemplate

# template = """Question: {question}

# Answer: Let's think step by step."""
# prompt = PromptTemplate.from_template(template)


# # model_name = "deepseek-ai/deepseek-llm-7b-chat"
# model_name = "meta-llama/Llama-2-7b-chat-hf"  # Adjust model if needed

# llm = HuggingFacePipeline.from_model_id(
#             model_id=model_name,
#             task="text-generation",
#             device=-1,  # -1 for CPU
#             # device_map="cuda" if torch.cuda.is_available() else "cpu",
#             # batch_size=2,  # adjust as needed based on GPU map and model size.
#             # device=0,  # replace with device_map="auto" to use the accelerate library.
#             # use_auth_token=True,
#             pipeline_kwargs={"max_new_tokens": 10},
# )

# gpu_chain = prompt | llm

# question = "What is electroencephalography?"

# print(gpu_chain.invoke({"question": question}))


from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat = ChatHuggingFace(llm=llm, verbose=True)