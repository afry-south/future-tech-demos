import streamlit as st

from haystack.preprocessor.utils import convert_files_to_dicts
from haystack.reader import TransformersReader
from haystack.document_store import InMemoryDocumentStore
from haystack.preprocessor import PreProcessor
from haystack.retriever.sparse import TfidfRetriever
from haystack.retriever.dense import DensePassageRetriever
from haystack.pipeline import ExtractiveQAPipeline


def load_retriever() -> DensePassageRetriever:
    document_store = InMemoryDocumentStore()

    docs = convert_files_to_dicts("qa-sweden/books")

    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=True,
        split_overlap=0,
        language="sv",
    )

    docs = processor.process(docs)

    document_store.write_documents(docs)

    return TfidfRetriever(document_store=document_store)


def load_reader() -> TransformersReader:
    return TransformersReader(
        model_name_or_path="KB/bert-base-swedish-cased-squad-experimental", use_gpu=-1
    )


@st.cache(allow_output_mutation=True)
def load_pipeline() -> ExtractiveQAPipeline:
    return ExtractiveQAPipeline(load_reader(), load_retriever())


pipeline = load_pipeline()


@st.cache
def predict(query: str, max_docs: int, max_answers: int):
    return pipeline.run(
        query=query,
        params={"Retriever": {"top_k": max_docs}, "Reader": {"top_k": max_answers}},
    )
