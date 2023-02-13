import openai
import streamlit as st
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import SimpleSequentialChain
openai.api_key = ...
st.set_page_config("Email Generator | AFRY X", page_icon="https://afry.com/themes/custom/afpt/favicon.ico")

@st.experimental_memo
def gen_img(prompt: str) -> str:
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    return response['data'][0]['url']

@st.experimental_memo
def gen_text(txt_prompt: str) -> str:
    llm = OpenAI(temperature=0.9, openai_api_key=openai.api_key)
    prompt = PromptTemplate(
        input_variables=["info"],
        template="Create a interesting introduction and then summarizing points in a slide based on a project you should build by the following information: {info}",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain.run(txt_prompt)

def main():
    c1,c2 = st.columns(2)
    c2.header("Slide Generator | AFRY X")
    c1.image("https://imgs.search.brave.com/8R3v_VlZ9vegVqfnkBVUZXLtRHuKBJQI3_5V4eodCpA/rs:fit:1200:386:1/g:ce/aHR0cHM6Ly9pbWcu/bGVkc21hZ2F6aW5l/LmNvbS9maWxlcy9i/YXNlL2VibS9sZWRz/L2ltYWdlLzIwMTkv/MTIvQWZyeV9sb2dv/X2Jhc2UuNWRlODE0/MDFjNDkwOS5wbmc_/YXV0bz1mb3JtYXQm/Zml0PW1heCZ3PTEy/MDA", width=200)

    with st.form("generation-form"):
        img_prompt = st.text_input("Generate Image")
        text_prompt = st.text_input("Generate Slide Based on")

        submitted = st.form_submit_button()

    if submitted:
        img_url = gen_img(img_prompt)
        text = gen_text(text_prompt)

        c1,c2 = st.columns(2)
        with c2:
            st.image(img_url, caption=img_prompt, width=400)
            st.write(f"[source]({img_url})")
        with c1:
            st.write(text)
    else:
        st.write("Submit something!")

if __name__ == '__main__':
    main()
