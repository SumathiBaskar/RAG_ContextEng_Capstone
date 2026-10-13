"""Capstone Checkpoint 4.1 — Advanced Retrieval Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 4.1 — Advanced Retrieval Implementation
# **MO-LLM Module 4/Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This checkpoint advances your capstone system from a *single-pass* retriever to an
# **advanced retriever**. You will use the baseline retrieval system you built in
# Checkpoint 2.1 and evaluated in Checkpoint 3.1 to identify where single-pass retrieval
# falls short and apply the Module 4 techniques to fix it:
#
# - **Lab 4.1 — multistep retrieval (query decomposition):** Rewrite a complex
#   question into focused sub-queries, retrieve for each, and merge the results.
# - **Lab 4.2 — graph-based retrieval:** Organize your corpus as a graph of
#   relationships (citations, shared topics, links, same author/entity) and pull in
#   *related* documents a keyword/vector search would miss.
#
# You then **measure the improvement** against your 3.1 baseline. The graded
# deliverable is a **500-750 word written report** (final section). This script is a
# runnable demonstration of both techniques on a tiny sample corpus so you can see
# them work before adapting them to your real system.
#
# **Learning outcomes (Module 4):**
# 1. Identify common retrieval failures — e.g., complex multistep questions and too
#    much loosely-related context degrading the answer.
# 2. Explain *why* those failures occur with single-pass retrieval.
# 3. Implement an advanced retrieval strategy (query decomposition and/or graph
#    traversal) that addresses them.
# 4. Measure the improvement against your baseline using your 3.1 evaluation set.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Natural relationships to exploit in a graph |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | citations, shared authors, shared topics/keywords, "published-before" |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | hyperlinks between articles, shared categories, mentioned entities |
#
# Multistep decomposition shines on cross-document questions ("Compare X and Y,"
# "how did an idea evolve"); graph retrieval shines when the answer needs documents
# that are *related* to a hit but don't themselves match the query terms.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv networkx`
# 3. Use the OpenRouter API key provided for this course. This checkpoint uses
#   the `openai/gpt-5.4-mini` model, with usage covered by course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This script runs on a tiny built-in sample corpus, so you do **not** need your full
# dataset indexed to complete it. You *will* refer to your real 2.1 system and 3.1
# evaluation results when you write the report.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import os
import re
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from pathlib import Path
from abc import ABC, abstractmethod

import networkx as nx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader # pyright: ignore[reportMissingImports]
from langchain_openai import OpenAIEmbeddings

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # Latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
LOG_PATH = Path.cwd() / "checkpoint_4_1_responses.log"

TOP_K = 5
EMBEDDING_MODEL = "openai/text-embedding-3-small"
NUM_RETRIEVED = 2        # Research Paper extracts sent to the LLM as context.
CANDIDATE_POOL = 10        # Candidates pulled from EACH retriever before fusion.
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 400


# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

SYSTEM_PROMPT = """You are a helpful assistant for Research Papers. \
You answer questions by drawing information exclusively from the research papers \
provided to you as context."""

DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a document retrieval system. "
    "Break the user's question into 2-4 focused sub-queries that together cover "
    "everything needed to answer it. Each sub-query should target a distinct aspect. "
    'Return ONLY a JSON array of strings, e.g., ["sub-query 1", "sub-query 2"].'
)

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using only the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at "
            "https://openrouter.ai/keys, put it in a .env file next to this "
            "script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )



def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(
            f"[{ts}] {label}\n"
            f"{text}\n"
            f"{'-' * 72}\n"
            )


class BaseRetriever(ABC):
    def __init__(self, llm_model: str = LLM_MODEL):
        self._llm = ChatOpenAI(
            model=llm_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
        self._history = [SystemMessage(content=SYSTEM_PROMPT)]

    @abstractmethod
    def retrievedContext(self, query: str) -> str: ...

    def _build_user_message(self, query: str, context: str) -> str:
        return (
            f"Research Paper excerpts:\n\n"
            f"{context}\n\n"
            f"Question: {query}"
        )

    def query(self, question: str) -> str:
        context = self.retrievedContext(question)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=self._build_user_message(question, context)),
        ]
        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)



#------- Stopwords to ignore----------#
_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
    "about", "above", "below", "between", "through", "during", "before", "after",
    "under", "over", "around", "along", "across", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "as", "if", "up", "out", "not", "no",
}

def tokenize(text: str) -> list[str]:
    return [
        t 
        for t in re.findall(r"[a-z0-9]+", text.lower()) 
        if t not in _STOPWORDS]

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )

def extract_pdf_documents(researchpapers_dir: str) -> list[Document]:
    documents = []

    pdf_paths = sorted(
        Path(researchpapers_dir).glob("*.pdf")
    )

    if not pdf_paths:
        raise RuntimeError(
            f"No PDF files were found in: {researchpapers_dir}"
        )

    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(str(pdf_path))

            pages = []

            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                text = re.sub(r"\s+", " ", text).strip()

                if text:
                    pages.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_number,
                            },
                        )
                    )

            documents.extend(pages)

        except Exception as exc:
            print(f"Could not process {pdf_path.name}: {exc}")

    return documents

def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks

def build_or_load_db(
    researchpapers_dir: str,
    chroma_dir: str = CHROMA_DIR,
) -> Chroma:

    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing vector DB from {chroma_dir}/")

        return Chroma(
            persist_directory=chroma_dir,
            embedding_function=get_embeddings(),
        )

    print("Extracting PDF text...")
    page_documents = extract_pdf_documents(researchpapers_dir)

    print(f"Extracted {len(page_documents)} pages.")
    print("Splitting papers into chunks...")

    chunks = chunk_documents(page_documents)

    print(f"Created {len(chunks)} chunks.")
    print("Building vector DB...")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=chroma_dir,
    )

    print(f"Vector DB created in {chroma_dir}/")
    return db


def _normalize(scores: list[float], invert: bool = False) -> list[float]:
    """Min-max scale a list of scores to [0, 1]. If invert is True, flip the scores 
    so a LOW raw value (e.g., a small vector distance = very similar) becomes a HIGH 
 normalized score."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)  # all equal → neutral
    norm = [(s - lo) / (hi - lo) for s in scores]
    return [1.0 - n for n in norm] if invert else norm

class HybridRetriever(BaseRetriever):
    def __init__(self,chunks: list[Document],db: Chroma,**kwargs,):
        super().__init__(**kwargs)
        self._chunks = chunks
        self._db = db
        self._bm25 = BM25Okapi([tokenize(chunk.page_content) for chunk in chunks])
        self._db = db
        print(f"Hybrid retriever ready over {len(chunks)}  research paper chunks.")

    def _bm25_topk(self, query: str, k: int) -> list[tuple[int, Document, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [
            (
                i,
                self._chunks[i],
                float(scores[i]),
            )
            for i in indices
        ]

    def _vector_topk(self, query: str, k: int) -> list[tuple[str, Document, float]]:
        results = self._db.similarity_search_with_score(query, k=k)
        output = []

        for document, score in results:
            chunk_id = document.metadata.get(
                "chunk_id",
                f"{document.metadata.get('source', 'unknown')}_"
                f"{document.metadata.get('page', 'unknown')}",
            )

            output.append(
                (
                    str(chunk_id),
                    document,
                    float(score),
                )
            )

        return output

    def getTopK(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Fuse the two retrievers: Pull a candidate pool from each, normalize, and
        combine with a weighted sum. BM25 is higher-is-better and vector distance is
        lower-is-better, so the vector side is normalized with invert=True. A doc
        missing from one retriever simply contributes 0 from that side."""
        bm25_results = self._bm25_topk(query, CANDIDATE_POOL)
        vector_results = self._vector_topk(query, CANDIDATE_POOL)

        content_by_id: dict[str, Document] = {}
        bm25_scores: dict[str, float] = {}
        vector_scores: dict[str, float] = {}

        if bm25_results:
            raw_scores = [
                score
                for _, _, score in bm25_results
            ]
            normalized_scores = _normalize(
                raw_scores
            )
            for (
                (index, document, _),
                normalized,
            ) in zip(
                bm25_results,
                normalized_scores,
            ):
                chunk_id = str(
                    document.metadata.get(
                        "chunk_id",
                        index,
                    )
                )
                content_by_id[chunk_id] = document
                bm25_scores[chunk_id] = normalized

        if vector_results:
            raw_scores = [
                score
                for _, _, score in vector_results
            ]
            normalized_scores = _normalize(
                raw_scores,
                invert=True,
            )
            for (
                (chunk_id, document, _),
                normalized,
            ) in zip(
                vector_results,
                normalized_scores,
            ):
                content_by_id[chunk_id] = document
                vector_scores[chunk_id] = normalized

        fused = []
        for chunk_id, document in content_by_id.items():

            score = (
                WEIGHT_BM25
                * bm25_scores.get(chunk_id, 0.0)
                +
                WEIGHT_VECTOR
                * vector_scores.get(chunk_id, 0.0)
            )

            fused.append(
                (
                    document,
                    score,
                )
            )
        fused.sort(
            key=lambda item: item[1],
            reverse=True,
        )
        return fused[:k]


    def retrievedContext(
        self,
        query: str,
    ) -> str:
        
        results = self.getTopK(
            query,
            NUM_RETRIEVED,
        )
        context_parts = []
        for rank, (document, score) in enumerate(
            results,
            start=1,
        ):
            source = document.metadata.get(
                "source",
                "unknown",
            )

            page = document.metadata.get(
                "page",
                "unknown",
            )
            context_parts.append(
                f"[Result {rank} | "
                f"Source: {source} | "
                f"Page: {page} | "
                f"Score: {score:.4f}]\n"
                f"{document.page_content}"
            )

        return "\n\n---\n\n".join(
            context_parts
        )

    def retrieve_with_scores(
        self,
        query: str,
    ) -> list[tuple[Document, float]]:

        return self.getTopK(
            query,
            NUM_RETRIEVED,
        )

def my_representative_queries() -> list[str]:
        MY_PROMPTS =[
        
        "What were the challenges of sketching and what were the benefits of streambit methodology ?",
        "ways to synthesize a program",
        "Is sketching at different levels of abstraction a challenge ?",
        "Explain low-tilt approximation",
        "What was the optimization proposed by program synthesis community ?",
        "Explain how policy agnostic programming helped in dynamic information flow for databased backed applications ?"
    ]
        return MY_PROMPTS  

def run(
    researchpapers_dir: str,
) -> None:
    print(
        f"Checkpoint 2.1 — baseline retrieval | "
        f"scenario: {SCENARIO}\n"
    )
    db = build_or_load_db(
        researchpapers_dir,
        CHROMA_DIR,
    )
    page_documents = extract_pdf_documents(
        researchpapers_dir
    )
    chunks = chunk_documents(
        page_documents
    )
    retriever = HybridRetriever(
        chunks=chunks,
        db=db,
    )
    queries = my_representative_queries()
    llm = make_llm()
    for i, query in enumerate(
        queries,
        start=1,
    ):

        print("=" * 72)
        print(f"ORIGINAL QUERY {i}: {query}\n")
        subqueries = decompose_query(llm,query)
        print(f"DECOMPOSED ORIGINAL QUERY INTO: \n {subqueries}")
        log(
            f"\nORIGINAL QUERY {i}: {query}\n",
            f"DECOMPOSED ORIGINAL QUERY INTO: \n {subqueries}",
            )

        for index, subquery in enumerate(subqueries, start=1):

            results = retriever.retrieve_with_scores(
                subquery
            )
            if not results:
                print(
                    "No relevant chunks were retrieved."
                )
                continue
            print("RETRIEVED RESULTS:")
            for rank, (document, score) in enumerate(
                results,
                start=1,
            ):
                source = document.metadata.get(
                    "source",
                    "unknown",
                )
                page = document.metadata.get(
                    "page",
                    "unknown",
                )
                print(
                    f"\n[{rank}] "
                    f"{source} | "
                    f"page {page} | "
                    f"score={score:.4f}"
                )
                print(
                    document.page_content[:1500]
                )
            answer = retriever.query(
                subquery
            )
            print("\nANSWER:")
            print(answer)
            retrieved_text = "\n\n".join(
                (
                    f"{document.metadata.get('source', 'unknown')} "
                    f"(page {document.metadata.get('page', 'unknown')}) "
                    f"score={score:.4f}\n"
                    f"{document.page_content}"
                )
                for document, score in results
            )
            log_response(
               
                f"{query}",
                f"{subquery}",
                f"{retrieved_text}",
                f"{answer}",

            )
        print("=" * 72)
        print("Done.")


def log_response(originalquery: str, subquery: str, retrievedresult: str, answer: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    entry = (
        f"{'-' * 72}\n"
        f"[{ts}] ORIGINAL QUERY: {originalquery}\n"
        f"SUBQUERY:   {subquery}\n"
        f"RETRIEVED RESULTS: {retrievedresult}\n"
        f"ANSWER: {answer}\n"
        f"{'-' * 72}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(entry)


# %% [markdown]
# ## Step 2 — Multistep retrieval (provided, adapted from Lab 4.1)
#
# `decompose_query` asks the LLM to split the question into sub-queries. For each
# sub-query, we retrieve the top documents and **sum** each document's score across
# the sub-queries. A document relevant to several parts of the question rises to the
# top. (In your real system, replace `baseline_retrieve` with your 2.1 retriever.)

# %%
def decompose_query(llm: ChatOpenAI, query: str) -> list[str]:
    messages = [SystemMessage(content=DECOMPOSE_SYSTEM), HumanMessage(content=query)]
    raw = llm.invoke(messages).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("["): raw.rfind("]") + 1] if "[" in raw else raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return [query]

# %% [markdown]
# ## Step 5 — Run baseline vs. advanced and capture the evidence
#
# This demonstration compares single-pass retrieval with the two advanced retrieval strategies 
# using the sample corpus and a representative query. It also logs the result. Read the
# retrieved sets: The advanced strategies should surface related documents that the
# baseline misses. Then run the same comparison in your real system for your report.

# %%

researchpapers_dir = sys.argv[1] if len(sys.argv) > 1 else "ResearchPapers"
run(researchpapers_dir)


