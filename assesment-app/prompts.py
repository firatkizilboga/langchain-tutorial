from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate


llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.9, max_tokens=10000)

knowledge_prompt = PromptTemplate(
    input_variables=["profession"],
    template="""Please provide 3 concepts/skills that a {profession} SHOULD know, consider their expected experience. NO EXPLANATIONS."""
)

knowledge_chain = LLMChain(
    prompt=knowledge_prompt,
    llm=llm,
    output_key="knowledge"
)


exam_prompt = PromptTemplate(
    input_variables=["profession", "knowledge"],
    template="""Please create a 3 question exam that a candidate for the position: {profession}.
The exam should test the following concepts/skills: {knowledge}
The questions can be multiple choice (four choices) or free form.
Include at least 1 free form question.
Example of a multiple choice question:
Q1 : What is the capital of France?
    a) Paris
    b) London
    c) Berlin
    d) Madrid
Correct answer is a) Paris
Example of a free-text question:
Q2) Why is the sky blue?
    Correct answer is: "The sky appears blue because of a phenomenon called Rayleigh scattering. When sunlight passes through the Earth's atmosphere, the shorter blue and violet wavelengths of light are scattered more by the nitrogen and oxygen molecules in the air. Our eyes are more sensitive to blue light, so we perceive the scattered blue light, making the sky appear blue."
""")

exam_chain = LLMChain(
    prompt=exam_prompt,
    llm=llm,
    output_key="exam"
)

json_prompt = PromptTemplate(
    input_variables=["exam"],
    template="""
Please extract the questions in the given text below, output format SHOULD be a python list of dictionaries here is one example:
EXAMPLE
INPUT
Q1 : What is the capital of France?
        a) Paris
        b) London
        c) Berlin
        d) Madrid
    Correct answer is a) Paris
Q2) Why is the sky blue?
    Correct answer is: "The sky appears blue because of a phenomenon called Rayleigh scattering. When sunlight passes through the Earth's atmosphere, the shorter blue and violet wavelengths of light are scattered more by the nitrogen and oxygen molecules in the air. Our eyes are more sensitive to blue light, so we perceive the scattered blue light, making the sky appear blue."
OUTPUT
[
    {{
        "question": "What is the capital of France?",
        "is_freeform": False,
        "a" : "Paris",
        "b" : "London",
        "c" : "Berlin",
        "d" : "Madrid",
        "correct_answer": "a"
    }},
    {{
        "question": "Why is the sky blue?",
        "is_freeform": True,
        "a": "",
        "b": "",
        "c": "",
        "d": "",
        "correct_answer": "The sky appears blue because of a phenomenon called Rayleigh scattering. When sunlight passes through the Earth's atmosphere, the shorter blue and violet wavelengths of light are scattered more by the nitrogen and oxygen molecules in the air. Our eyes are more sensitive to blue light, so we perceive the scattered blue light, making the sky appear blue."
    }}
]
END OF EXAMPLE

{exam}""",
)



json_chain = LLMChain(
    prompt=json_prompt,
    llm=llm,
    output_key="json"
)

application_chain = SequentialChain(
    chains=[knowledge_chain,exam_chain,json_chain],
    input_variables=["profession"],
    output_variables=["json"],
    verbose=True
)



freeform_scoring_prompt = PromptTemplate(
    input_variables=["question", "answer", "user_answer"],
    template="""Given the following question and its correct answer, please provide a score out of 10 to the user's answer.
EXAMPLE
INPUT
Question: Why is the sky blue?
Answer: The sky appears blue because of a phenomenon called Rayleigh scattering. When sunlight passes through the Earth's atmosphere, the shorter blue and violet wavelengths of light are scattered more by the nitrogen and oxygen molecules in the air. Our eyes are more sensitive to blue light, so we perceive the scattered blue light, making the sky appear blue.
Users Answer: The sky appears blue because the sunlight contains blue pigment. When the sun shines, it releases tiny blue particles that spread throughout the atmosphere, giving the sky its blue color. These blue particles then reflect the sunlight back to our eyes, creating the blue sky phenomenon.
OUTPUT
2
END OF EXAMPLE
Question: {question}
Answer: {answer}
Users Answer: {user_answer}
"""
)

freeform_scoring_chain = LLMChain(
    prompt=freeform_scoring_prompt,
    llm=llm,
    output_key="score"
)

extraction_examples = [
    {
        "question": "Score: 2/10 The user's answer is incorrect as it talks about the sun releasing blue pigment, which is not true. The explanation given is also not scientifically accurate. Therefore, the score is 2/10.",
        "answer": "2",
    },
    {
        "question": "Score: 10/10 The user's answer is correct as it talks about the sun releasing blue particles, which is true. The explanation given is also scientifically accurate. Therefore, the score is 10/10.",
        "answer": "10",
    },
    {
        "question": "4.3",
        "answer": "4",
    },
    {
        "question": "N/A",
        "answer": "0",
    },
    {
        "question": "I couldn't find the score",
        "answer": "0",
    },
    {
        "question": "Here is the score: 5",
        "answer": "5",
    }
]
extraction_example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="""Question: {question}\nAnswer: {answer}\n"""
)

extraction_prompt = FewShotPromptTemplate(
    examples=extraction_examples, 
    example_prompt=extraction_example_prompt, 
    suffix="Here is the score: {score}", 
    input_variables=["score"]
)

extraction_chain = LLMChain(
    llm=llm,
    prompt=extraction_prompt,
)

handle_freeform_chain = SequentialChain(
    chains=[freeform_scoring_chain, extraction_chain],
    input_variables=["question", "answer", "user_answer"],
    output_variables=["score"],
    verbose=True
)


def main():
    profession = input("Enter the profession: ")
    result = application_chain.run({"profession": profession})
    print(eval(result))

if __name__ == "__main__":
    main()