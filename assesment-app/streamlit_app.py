import streamlit as st
from prompts import application_chain, handle_freeform_chain
import uuid


st.title("Quiz App")

class QuestionObject():
    @staticmethod
    def from_dict(dictionary: dict) -> 'QuestionObject':
        return QuestionObject(
            question=dictionary.get('question'),
            is_freeform=dictionary.get('is_freeform'),
            a=dictionary.get('a'),
            b=dictionary.get('b'),
            c=dictionary.get('c'),
            d=dictionary.get('d'),
            correct_answer=dictionary.get('correct_answer')
        )
    
    def __init__(
        self,
        question: str,
        is_freeform: bool,
        a: str,
        b: str,
        c: str,
        d: str,
        correct_answer: str):
        self.question = question
        self.is_freeform = is_freeform
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.correct_answer = correct_answer
        self.uid = uuid.uuid4()
    

    def render(self):
        st.write(self.question)
        if self.is_freeform:
            self.answer = st.text_input("Answer", key = str(self.uid))
        else:
            # Use session state to store the answer across reruns
            if f'answer_{self.uid}' not in st.session_state:
                st.session_state[f'answer_{self.uid}'] = None
            st.session_state[f'answer_{self.uid}'] = st.radio("Answer", [self.a, self.b, self.c, self.d], key = str(self.uid))
            self.answer = st.session_state[f'answer_{self.uid}']

    def score(self):
        if self.is_freeform:
            return self.handle_freeform()

        if self.answer == getattr(self, self.correct_answer):
                return 10
        
        return 0
                

    def handle_freeform(self):
        score = handle_freeform_chain.run(
            question=self.question,
            answer=self.correct_answer,
            user_answer=self.answer,
        )

        try:
            return float(score)
        except:
            return 0

def main():
    position = st.text_input("What is the position that you want to create an exam for?")

    # Check if 'exam' is in the session state
    if 'exam' not in st.session_state:
        st.session_state.exam = None

    if st.button("GO!"):
        st.session_state.exam = [QuestionObject.from_dict(question) for question in eval(application_chain.run(profession=position))]

    # Render the questions if they exist
    if st.session_state.exam is not None:
        for question in st.session_state.exam:
            question.render()

        submit = st.button("Submit")
        if submit:
            score = 0
            for question in st.session_state.exam:
                score += question.score()

            st.write(f"Your score is {score}")

if __name__ == "__main__":
    main()