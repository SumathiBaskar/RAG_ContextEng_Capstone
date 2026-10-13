
"""Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System
# **MO-LLM Module 5 -  Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This is the final build step of your capstone. You will turn the retrieval system you've
# developed into an **agent-based RAG system**. Instead of a fixed pipeline, an agent
# decides at each step whether it has enough information, what to retrieve next, which
# tool to use, or whether to answer. This activity allows you to apply the Module 5 labs (Lab 5.1's agentic
# retriever and Lab 5.2's tool-using agent) to your capstone scenario.
#
# The graded deliverable is your completed Capstone Checkpoint 5.1 worksheet. This script is a
# runnable demonstration of an agentic loop on a tiny sample corpus so you can see the
# decision-making before adapting it to your full system.
#
# **Learning outcomes (Module 5):**
# 1. Build an agent-based system that integrates retrieval and external tools.
# 2. Use system prompts to guide agent behavior and decision-making.
# 3. Design workflows that coordinate retrieval, reasoning, and tool use in a RAG system.
# 4. Evaluate an agentic workflow against a fixed retrieval pipeline.
# 5. Design workflows that coordinate retrieval, reasoning, and tool use within a RAG system.
# 6. Build an agent-based system that integrates retrieval and external tools to complete user tasks.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Useful agent tools/actions |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs | semantic search; "find papers by author/year"; "follow citations"; ask a clarifying question |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles | semantic search; "find by category"; "follow links"; ask a clarifying question |
#
# An agent shines when one query isn't enough — it can search, look at what it found,
# and decide to search again (or use a different tool) before answering.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Get a free OpenRouter key at <https://openrouter.ai/keys>. The labs use the paid
#    `openai/gpt-5.4-mini` model, which is covered by the course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). The built-in corpus is deliberately minimal: It only demonstrates
# the agent control flow, not retrieval quality, and it is unrelated to your capstone:
# Howerver, your design should target your own capstone corpus. Your full agent is what you describe
# in the written submission.

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
LLM_MODEL = "openai/gpt-5.4-mini"
TEMPERATURE = 0.2
MAX_STEPS = 3
LOG_PATH = Path.cwd() / "checkpoint_5_1_agent.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

TOP_K = 5
EMBEDDING_MODEL = "openai/text-embedding-3-small"
NUM_RETRIEVED = 2        # Research Paper extracts sent to the LLM as context.
CANDIDATE_POOL = 10        # Candidates pulled from EACH retriever before fusion.
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 400

DECIDE_SYSTEM = (
   "You are an agent retrieving from a small document collection. Given the question, "
       "the queries already run, and the documents found so far, decide what to do next. "
       'Respond with ONLY a JSON object: {"done": true|false, "new_queries": ["..."], '
       '"reasoning": "..."}. Set done=true when you have enough to answer; otherwise give '
       "1-2 new_queries targeting what is still missing (do not repeat past queries)."
    
)
ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided documents, "
    "quoting where you can. If they do not contain the answer, say so."
)

SYSTEM_PROMPT = """You are a helpful assistant for Research Papers. \
You answer questions by drawing information exclusively from the research papers \
provided to you as context."""

# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at https://openrouter.ai/keys, "
            "put it in a .env file next to this script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE,
                      api_key=check_api_key(), base_url=OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")

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
                    document,score,
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
    ) -> list[str]:
        
        results = self.getTopK(
            query,
            NUM_RETRIEVED,
        )
        context_parts = []
        for rank, (document, score) in enumerate(
            results,
            start=1,
        ):
            context_parts.append(
                f"{document.page_content}"
            )
        return "\n\n---\n\n".join(
            context_parts
        )


def decide(llm: ChatOpenAI, question: str, collected: dict[str, str], executed: list[str]) -> dict:
    docs = "\n".join(f"{item}" for item in collected) or "(none yet)"
    user = f"Question: {question}\n\nQueries run: {executed or '(none)'}\n\nDocuments so far:\n{docs}"
    raw = llm.invoke([SystemMessage(content=DECIDE_SYSTEM), HumanMessage(content=user)]).content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        d = json.loads(raw)
        return {"done": bool(d.get("done", True)), "new_queries": d.get("new_queries", []) or [],
                "reasoning": d.get("reasoning", "")}
    except (json.JSONDecodeError, ValueError):
        return {"done": True, "new_queries": [], "reasoning": "parse-fail -> stop"}


def agentic_answer(llm: ChatOpenAI, question: str, results: list[str]) -> str:
    """Provided: a minimal agentic loop — retrieve, decide whether to continue, repeat."""
    collected: dict[str, str] = {}
    executed: list[str] = []
    pending = [question]
    for step in range(MAX_STEPS):
        for q in pending:
            for index, res in enumerate(results):
                collected[index] = res
            executed.append(q)
        d = decide(llm, question, collected, executed)
        print(f"  step {step + 1}: \n\n -> retrievel done={d['done']}\n\n reasoning= ({d['reasoning'][:250]})\n\n new queries=({d['new_queries']})")
        if d["done"] or not d["new_queries"]:
            break
        pending = d["new_queries"]
    context = "\n\n".join(f"[{i}] {collected[i]}" for i in collected)
    return llm.invoke([SystemMessage(content=ANSWER_SYSTEM),
                       HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}")]).content


# %% [markdown]
# ## Step 2 — Your agent design (TODO)
#
# Design the agent you will build for **your** scenario. Return a dictionary with the keys
# below — this is the plan you implement in your real system and describe in the report.

# %%
def my_agent_plan() -> dict[str, Any]:
        return {
            "tools": ["semantic_search","find by citations", "clarify", "answer"],
            "stop_condition": "the retrieved passages cover every part of the question",
            "system_prompt_idea": ["Search first; clarify if the request is ambiguous; ",
                                  "answer once the retrieved text supports a grounded reply."],
            "test_tasks": [
                "What are the different benchmarks used to test REAS ", 
            ],
        }

# %% [markdown]
# ## Step 3 — Run the agentic loop and capture the evidence
#
# Runs the provided agentic loop on a multistep question (watch it retrieve, decide,
# and retrieve again), then prints your plan. Reproduce this in your real system for the
# report and compare it against your fixed Checkpoint 2.1 retriever on the same task.

# %%
def run(researchpapers_dir: str,) -> None:
    llm = make_llm()
    question = "Give me the list of all research paper names that is related to program synthesis"
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
    results = retriever.retrievedContext(
                    question
    )
    if not results:
        print(
            "No relevant chunks were retrieved."
        )
    print(f"Checkpoint 5.1 — agentic RAG Capstone")
    print(f"Question: {question}\n")
    answer = agentic_answer(llm, question, results)
    print(f"\nAgent answer:\n{answer}\n")
    log("AGENTIC", f"Q: {question}\nA: {answer}")
    """try:
        print("Your agent plan:")
        print(json.dumps(my_agent_plan(), indent=2))
    except NotImplementedError as e:
        print(f"[my_agent_plan not done yet] {e}")
    """
    print("=" * 72)
    print("Done.")

researchpapers_dir = sys.argv[1] if len(sys.argv) > 1 else "ResearchPapers"
run(researchpapers_dir)

# %% [markdown]
# ## Step 4 — Your written submission (the graded deliverable)
#
# The deliverable is a **written submission** (suggested length: 500-750 words). Cover the following:
#
# 1. **System overview**: State your scenario and the retrieval system you built (2.1–4.1).
# 2. **Agent design**: Describe the actions/tools your agent can take, and how a **system prompt**
#    guides it to choose among them (retrieve, use a tool, clarify, or answer).
# 3. **Workflow**: Explain how retrieval, reasoning, and tool use are coordinated across steps,
#    and how the agent decides it has enough information to answer.
# 4. **Evaluation**: Compare the agent-based system against your fixed Checkpoint 2.1/3.1
#    pipeline on a few representative tasks. Did multi-step decision-making improve the
#    answers? At what cost (extra model calls, latency)?
# 5. **Reflection**: When an agentic approach adds value vs. when a simpler pipeline is
#    better, and the limitations/trade-offs of increasingly autonomous systems.
#
# Include evidence (sample tasks, the agent's step-by-step decisions from a log, before/
# after comparison).
