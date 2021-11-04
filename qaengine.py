from typing import List
import streamlit as st

from haystack.file_converter.pdf import PDFToTextConverter
from haystack.reader import FARMReader, TransformersReader
from haystack.document_store import InMemoryDocumentStore
from haystack.preprocessor import PreProcessor
from haystack.retriever.sparse import TfidfRetriever
from haystack.pipeline import ExtractiveQAPipeline
from haystack.utils import print_answers

def load_retriever() -> TfidfRetriever:
    document_store = InMemoryDocumentStore()

    converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=['sv'])
    doc = converter.convert(file_path='omsverige_v7_se.pdf', encoding='UTF-8', meta=None)

    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=True,
        split_overlap=0,
        language='sv'
    )
    docs = processor.process(doc)

    document_store.write_documents(docs)
    
    return TfidfRetriever(document_store=document_store)

def load_reader() -> TransformersReader:
    return TransformersReader(model_name_or_path='susumu2357/bert-base-swedish-squad2', use_gpu=-1)

@st.cache
def load_pipeline() -> ExtractiveQAPipeline:
    return ExtractiveQAPipeline(load_reader(), load_retriever())

pipeline = load_pipeline()

@st.cache
def predict(query: str, max_docs: int, max_answers: int) -> List[dict[str, str]]:
    return pipeline.run(query=query, params={"Retriever": {"top_k": max_docs}, "Reader": {"top_k": max_answers}})
