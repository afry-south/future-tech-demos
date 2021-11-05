import os
import sys
import logging
import pandas as pd
import streamlit as st
from annotated_text import annotated_text

from qaengine import predict


def annotate_answer(answer, context):
    """ If we are using an extractive QA pipeline, we'll get answers
    from the API that we highlight in the given context"""
    start_idx = context.find(answer)
    end_idx = start_idx + len(answer)
    # calculate dynamic height depending on context length
    height = int(len(context) * 0.50) + 5
    annotated_text(context[:start_idx], (answer, "ANSWER", "#8ef"), context[end_idx:])


def main():
    # UI search bar and sidebar
    st.write("# AFRY Future Tech (IT Syd) QA Demo")
    st.sidebar.header("Variabler")
    top_k_reader = st.sidebar.slider("Max. antal svar", min_value=1, max_value=10, value=3, step=1)
    top_k_retriever = st.sidebar.slider(
        "Max. antal dokument från 'Retrievern'", min_value=1, max_value=20, value=10, step=1
    )
    # eval_mode = st.sidebar.checkbox("Evalueringsläge")
    debug = st.sidebar.checkbox("Visa debuginformation")

    st.text("""
    Exempelinput:
      Vem är Akka?
      Hur många samer bor i sverige?
      Vem är Smirre?
      Vem var Olof Palme?
      Hur många landskap har sverige?
      Vem upptäckte DNA?
      När grundades disney?
    """)

    # Search bar
    question = st.text_input("Var snäll skriv din fråga:", value="När var första VM?")
   
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
                st.markdown(result["context"])
            
            if result['answer'] or result['context']:
                st.write("**Relevans:** ", result["score"], "**Källa:** ", result['meta']['name'])
            st.write("___")
        if debug:
            st.subheader("REST API JSON-respons")
            st.write(raw_json)

main()