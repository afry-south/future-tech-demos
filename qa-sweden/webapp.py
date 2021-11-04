import os
import sys


# https://code.visualstudio.com/docs/remote/containers -- DOCKER

import logging
import pandas as pd
import streamlit as st
from annotated_text import annotated_text

from qaengine import predict

# streamlit does not support any states out of the box. On every button click, streamlit reload the whole page
# and every value gets lost. To keep track of our feedback state we use the official streamlit gist mentioned
# here https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
# import SessionState

# from utils import feedback_doc, haystack_is_ready, retrieve_doc, upload_doc

# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = "Vad får jag slänga i toaletten?"


def annotate_answer(answer, context):
    """ If we are using an extractive QA pipeline, we'll get answers
    from the API that we highlight in the given context"""
    start_idx = context.find(answer)
    end_idx = start_idx + len(answer)
    # calculate dynamic height depending on context length
    height = int(len(context) * 0.50) + 5
    annotated_text(context[:start_idx], (answer, "ANSWER", "#8ef"), context[end_idx:])


def show_plain_documents(text):
    """ If we are using a plain document search pipeline, i.e. only retriever, we'll get plain documents
    from the API that we just show without any highlighting"""
    st.markdown(text)


def random_questions(df):
    """
    Helper to get one random question + gold random_answer from the user's CSV 'eval_labels_example'.
    This can then be shown in the UI when the evaluation mode is selected. Users can easily give feedback on the
    model's results and "enrich" the eval dataset with more acceptable labels
    """
    random_row = df.sample(1)
    random_question = random_row["Question Text"].values[0]
    random_answer = random_row["Answer"].values[0]
    return random_question, random_answer


def main():
    # Define state
    #state_question = SessionState.get(
    #    random_question=DEFAULT_QUESTION_AT_STARTUP, random_answer="", next_question="false", run_query="false"
    #)

    # Initialize variables
    eval_mode = False
    random_question = DEFAULT_QUESTION_AT_STARTUP
    # eval_labels = os.getenv("EVAL_FILE", "eval_labels_example.csv")

    # UI search bar and sidebar
    st.write("# AFRY Future Tech (IT Syd) QA Demo")
    st.sidebar.header("Val")
    top_k_reader = st.sidebar.slider("Max. antal svar", min_value=1, max_value=10, value=3, step=1)
    top_k_retriever = st.sidebar.slider(
        "Max. antal dokument från 'Retrievern'", min_value=1, max_value=20, value=10, step=1
    )
    # eval_mode = st.sidebar.checkbox("Evalueringsläge")
    debug = st.sidebar.checkbox("Visa debuginformation")

    st.text("""
    Exempelinput:
      När fick kvinnor börja rösta i Sverige?
      Vart bor samerna?
      Hur gammal måste jag vara för att få dricka alkohol?
      Vad får jag slänga i toaletten?
      När blir man myndig?
      När var första världskriget?
      Vem var Olof Palme?
      hur fungerar det med arv i sverige?
      finns det tandvård i sverige?
    """)

    # Search bar
    question = st.text_input("Var snäll skriv din fråga:", value=random_question)
   
    raw_json_feedback = ""

    # Get results for query
    if len(question):
        with st.spinner(
            "🧠 &nbsp;&nbsp; Genomför neural sökning på dokument... \n "
            "Vill du optimera för hastighet eller pricksäkerhet? \n"
            "Spana in dokumentationen (/usage/optimization)"
        ):
            try:
                raw_json = predict(question, max_answers=top_k_reader, max_docs=top_k_retriever)
                results = raw_json['answers']
            except Exception as e:
                logging.exception(e)
                st.error("🐞 &nbsp;&nbsp; Ett fel inträffade under anroppet. Kontrollera log i konsol för att få mer detaljer.")
                return

        st.write("## Resultat:")

        # Make every button key unique
        count = 0
        
        # TODO potentially add Eval-Mode
        for result in results:
            if result["answer"]:
                annotate_answer(result["answer"], result["context"])
            elif result['context']:
                show_plain_documents(result["context"])
            
            if result['answer'] or result['context']:
                st.write("**Relevans:** ", result["score"], "**Källa:** ", result["document_id"])
            st.write("___")
        if debug:
            st.subheader("REST API JSON-respons")
            st.write(raw_json)

main()