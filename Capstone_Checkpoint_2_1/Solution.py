"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation
# **MO-LLM Module 2 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# In Checkpoint 1.1, you showed that a plain LLM can't reliably answer questions about
# your corpus. Now, you will **add retrieval**: Design a retrieval strategy for your scenario
# and build a **baseline retrieval system** that finds the most relevant documents for
# a query, so the model can ground its answers in them.
#
# This mirrors the Module 2 labs — keyword (BM25), vector (semantic), and hybrid
# retrieval — applied to your own capstone corpus. The graded deliverable is the completed 
# Capstone Checkpoint 2.1 worksheet, which includes your written responses and evidence of your 
# retrieval system implementation and testing. This script provides a small working example of 
# baseline retrieval. Use it to understand the retrieval workflow, then adapt the code to implement 
# and test a baseline retriever using your selected capstone dataset.
#
# **Learning outcomes (Module 2):**
# 1. Design a retrieval strategy appropriate for a given dataset and query type.
# 2. Implement and test a baseline retrieval system using structured and/or semantic
#    approaches.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1.
#
# | Scenario | Corpus | Retrieval considerations |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | long documents; you'll likely chunk them; questions often name a specific paper or compare papers. |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | many short-to-medium articles; questions name a figure/place or span several articles. |
#
# A good baseline is keyword (BM25), semantic (embeddings + vector search), or a
# hybrid of both — exactly what you built in Labs 1.2–2.2.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program. This checkpoint uses
#  the `openai/gpt-5.4-mini` model, with usage covered by the course credits. (this uses the paid gpt-5.4-mini chat model — covered by your course credits — and a keyword retriever, no embeddings).
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). Your real baseline (over your full corpus) is what you describe in
# the writeup.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader # pyright: ignore[reportMissingImports]


# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
TOP_K = 5
LOG_PATH = Path.cwd() / "checkpoint_2_1_retrieval.log"
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

SYSTEM_PROMPT = """You are a helpful assistant for Precision Paperclip Inc. \
You answer questions by drawing information exclusively from the company e-mails \
provided to you as context in each message.

Rules:
- If the answer can be found in the provided e-mails, answer clearly and concisely.
- If the provided e-mails do not contain enough information to answer the question, \
say so explicitly and do not speculate or use outside knowledge.
- Do not answer questions that are unrelated to the content of the provided e-mails."""


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this course, put it in a .env file next to this "
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

# %% [markdown]
# ## Step 2 — The baseline retriever (provided)
#
# A simple keyword-overlap retriever scores each document by how many query words
# it shares, and returns the top-k. This is the smallest possible baseline (a stand-in
# for the BM25 / vector / hybrid retriever you built in the Module 2 labs). The
# `answer` function then asks the LLM using only the retrieved documents.

# %%
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


# %% [markdown]
# ## Step 3 — Your representative queries (TODO)
#
# Submission item #2 asks for **3-5 representative queries** for your scenario and the
# results your system retrieves for each. Write those queries here. Some good ones include questions that:
#
# - Are answerable from **one** document (tests precision),
# - Need **several** documents (tests recall / aggregation),
# - Have wording that **differs** from the document's wording (i.e., tests whether
#   keyword vs. semantic retrieval matters for your corpus)
#
# Return a list of 3-5 query strings.

# %%
def my_representative_queries() -> list[str]:
    
    """
    What were the challenges of sketching and what were the benefits of streambit methodology ?,
    ways to synthesize a program",
    Is sketching at different levels of abstraction a challenge ?,
    Explain low-tilt approximation,
    What was the optimization proposed by program synthesis community ?,
    Explain how policy agnostic programming helped in dynamic information flow for databased backed applications ?
    """
    MY_PROMPTS =[
        
        "Give me the list of all research paper names that is related to program synthesis",
        "What are the different benchmarks used to test REAS?",
        " What problem does SAT solver solve in real world ?",
    ]
    return MY_PROMPTS


# %% [markdown]
# ## Step 4 — Run the baseline and capture the evidence
#
# This runs each query through the baseline retriever and the LLM, printing the
# retrieved document ids/scores and the grounded answer, and logging everything to
# `checkpoint_2_1_retrieval.log`. The retrieved documents from the output are the rest of the evidence for
# submission item #2.

# %%
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

    for i, query in enumerate(
        queries,
        start=1,
    ):

        print("=" * 72)
        print(f"QUERY {i}: {query}\n")
        results = retriever.retrieve_with_scores(
            query
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
            query
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
        log(
            f"QUERY {i}: {query}",
            f"retrieved={retrieved_text}\n\n"
            f"answer={answer}",
        )
    print("=" * 72)
    print(
        "Done."
        
    )



if __name__ == "__main__":

    researchpapers_dir = sys.argv[1] if len(sys.argv) > 1 else "ResearchPapers"
    run(researchpapers_dir)



# %% [markdown]
# ## Step 5 — Your written submission (the graded deliverable)
#
# Use your completed retrieval implementation and test results to complete the Capstone Checkpoint 2.1
# worksheet. In the worksheet, you will document your retrieval approach, provide evidence that your 
# system is functioning, include 3–5 representative queries and retrieved results, and reflect on where
# your approach performs well and where it struggles.  
 
# Save your completed Python file in the appropriate checkpoint folder in your GitHub repository. 
# Upload the completed worksheet only to the learning platform as your graded submission.
