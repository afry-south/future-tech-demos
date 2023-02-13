import streamlit as st
from gpt_index import GPTSimpleVectorIndex

st.set_page_config("AFRY X Document Assistant", page_icon="https://afry.com/themes/custom/afpt/favicon.ico",
                   layout="wide")

@st.experimental_memo
def load_index() -> GPTSimpleVectorIndex:
    return GPTSimpleVectorIndex.load_from_disk('index.json')


def main():
    st.image("https://www.symetri.se/media/opbeyksp/afry_logotyp_liggande_png-1.png")
    st.header("AFRY X | Document Assistant")

    st.file_uploader("Upload documents", disabled=True)

    previous_chat = st.write("""
    **Q:** How do I do X?
    **A:** Apply AI.
    """)
    query = st.text_area("Ask your question")
    index = load_index()
    index.query(query)


if __name__ == '__main__':
    main()
