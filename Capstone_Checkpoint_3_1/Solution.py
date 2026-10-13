"""Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.

This demonstration system is not your capstone system and 
does not use the Research Paper Navigator or Wikipedia corpus.
"""

# %% [markdown]
# # Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis
# **MO-LLM Module 3 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# #
# This mirrors **Lab 3.1** (an LLM-judge that scores answers pass/fail against grading
# notes), applied to your capstone system. 
# You have a baseline retrieval system from Checkpoint 2.1. In this checkpoint, 
# you will use a structured evaluation approach to measure baseline performance, diagnose strengths 
# and weaknesses, and examine whether the evaluation framework detects problematic outputs.
# The starter script includes a small demonstration corpus and retriever to illustrate the 
# evaluation workflow. Apply the same evaluation approach to your selected capstone scenario and 
# baseline retrieval system. The graded deliverable is the completed Capstone Checkpoint 3.1 worksheet.
#
# **Learning outcomes (Module 3):**
# 1. Define evaluation metrics that reflect the real-world performance requirements of an LLM-powered retrieval system.
# 2. Identify key variables that influence system performance during evaluation and development.
# 3. Use language models to support evaluation tasks while avoiding common pitfalls.
# 4. Evaluate the performance of a retrieval-augmented system during development using a structured evaluation framework.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario and baseline retriever** from Checkpoints 1.1 and 2.1.
#
# | Scenario | Corpus |
# |---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) |

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12.**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this course.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# Runs on a tiny built-in sample corpus, so you do not need to prepare your own dataset.
# It still requires an OpenRouter API key to run the LLM (it is not offline or free of API
# calls). You apply the same evaluation approach to your real system for the report.

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
LOG_PATH = Path.cwd() / "checkpoint_3_1_evaluation.log"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
NUM_RETRIEVED = 2        # Research Paper extracts sent to the LLM as context.
CANDIDATE_POOL = 10        # Candidates pulled from EACH retriever before fusion.
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 400
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)
JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Reply with exactly one word: 'pass' if the "
    "answer satisfies the grading notes, or 'fail' if it does not."
)



# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set . Use the OpenRouter API key "
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
# ## Demonstration corpus + retriever (provided to illustrate the evaluation workflow)

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
    
def my_representative_queries() -> list[str]:
    MY_PROMPTS =[
        
        "What were the challenges of sketching and what were the benefits of streambit methodology ?",
        "ways to synthesize a program",
        "Explain low-tilt approximation",
        "Is sketching at different levels of abstraction a challenge ?",
    ]
    return MY_PROMPTS



# %% [markdown]
# ## Step 2 — The evaluation metric (provided)
#
# A simple **LLM-judge** that returns pass/fail by checking an answer against grading
# notes — the same idea as Lab 3.1's DiscreteMetric, written directly here so the
# checkpoint needs no extra packages. Tuning this metric (stricter notes, a 'partial'
# level, a stronger judge model) is part of the diagnosis.

# %%
def judge(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> str:
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\nVerdict (pass/fail):"),
    ]
    verdict = llm.invoke(messages).content.strip().lower()
    return "pass" if "pass" in verdict else "fail"


# %% [markdown]
# ## Step 3 — Your evaluation set (TODO)
#
# Define the test set your evaluation runs on. Each item is a question plus
# **grading notes** — a short description of what a correct answer must contain (the
# judge checks the answer against these). Good evaluation sets include questions you
# expect to pass AND questions that probe known weaknesses.
#
# Return a list of 3-5 dicts: `{"question": "...", "grading_notes": "..."}`.

# %%
def my_eval_set() -> list[dict]:
    
    question_gradingnotes: list[dict] = [
    {"question": "What were the challenges of sketching and what were the benefits of streambit methodology ?", 
     "grading_notes": 
     "StreamBit system can be a good choice when developing prototypes of ciphers, because it allows for\n"
     "the code to be developed faster, and the code is actually of muc better quality than code produced \n"
     "by hand in a comparable amount of time. Given the same time budget, the ciphers developed in StreamBit\n"
     "ran 2.5×faster than ciphers coded in C.\n"
     "StreamBit methodology offers the following benefits: (1) it obviates the need to code low-level details \n"
     "thereby making developers more productive, (2) it rejects buggy sketches and guarantees correctness \n"
     "by construction, (3) it allows the programmer to rapidly test and evaluate various implementation ideas,\n"
     "even before they are known to be correct—without the fear of introducing bugs. In addition, \n"
     "the sketches are robust because they are devoid of low-level details. "},

    {"question": "ways to synthesize a program", 
     "grading_notes": 
     "Information flow control is important to securing the application-database boundary. \n"
     "This is because leaks often involve the results of computations on sensitive values, \n"
     "rather than sensitive values themselves. To reduce the opportunity for inadvertent leaks,\n"
     "we present a policy-agnostic approach"},

    {"question": "Explain low-tilt approximation", 
     "grading_notes": 
     "We introduce a parameter into the sampling algorithm, d, that parameter-izes q(·). \n"
     "The parameter d acts as a threshold, or cut-off, for the description length of a program;\n"
     "the distribution q(·) acts as though any program with description length exceeding "
     "can be encoded using d bits."}
    ]

    return question_gradingnotes


# %% [markdown]
# Run the demonstration code to understand the evaluation workflow. Then apply the same evaluation 
# design to your own capstone baseline system, using its retrieval and answer-generation functions. 
# Record results from your capstone system in the worksheet.
#
# ## Step 4 — Run the baseline evaluation
#
# Answers each question with the baseline retriever, scores it with the judge, and
# reports the pass rate. This is your baseline diagnosis: the failures are what you
# analyse in the report.

# %%
def run_evaluation(chunked_documents: list[Document],db: Chroma) -> None:
    passes = 0
    retriever = HybridRetriever(chunks=chunked_documents,db=db,)
    eval_set = my_eval_set()
    print(f"Evaluation started....")
    for i, item in enumerate(eval_set, 1):
        hits= retriever.retrieve_with_scores(item["question"])
        ans = retriever.query(item["question"]) if hits else "(no documents retrieved)"
        verdict = judge(llm, ans, item["grading_notes"])
        passes += verdict == "pass"
        print("=" * 72)
        print(f"Q{i}: {item['question']}")
        print(f"  retrieved={hits}  verdict={verdict.upper()}")
        print(f"  answer: {ans}")
        log(f"Q{i}: {item['question']}", f"retrieved={hits}\nverdict={verdict}\nanswer={ans}")
    print("=" * 72)
    print(f"Baseline pass rate: {passes}/{len(eval_set)}")



# %% [markdown]
# ## Step 5 — Validate the framework: can it catch a manipulated answer? (provided)
#
# A good evaluation framework must FAIL a wrong answer, not just pass good ones. This
# takes a question your corpus can answer, produces a correct answer, then feeds the
# judge a deliberately manipulated (false) answer — and checks that the judge flags it.
# This is your "evaluation framework validation" evidence for the report.

# %%


def validate_framework(chunked_documents: list[Document],db: Chroma) -> None:
    llm = make_llm()
    validation_question = "ways to synthesize a program"
    grading_benchmark = "sketching and constraint solving"
    retriever = HybridRetriever(chunks=chunked_documents,db=db,)
    good_answer =  retriever.query(validation_question) 
    manipulated_answer = "BM25 is a radio frequency technique."
    good_verdict = judge(llm, good_answer, grading_benchmark)
    manip_verdict = judge(llm, manipulated_answer, grading_benchmark)
    print("\n--- Framework validation ---")
    print(f"  correct answer   -> {good_verdict.upper()}   (expected PASS)")
    print(f"  manipulated answer -> {manip_verdict.upper()}   (expected FAIL)")
    print("  The framework works if it PASSES the correct answer and FAILS the manipulated one.")
    log("FRAMEWORK VALIDATION", f"good={good_verdict} manipulated={manip_verdict}")


if __name__ == "__main__":
    researchpapers_dir = sys.argv[1] if len(sys.argv) > 1 else "ResearchPapers"
    print(f"Checkpoint 3.1 — baseline evaluation  |  scenario: {SCENARIO}\n")
    llm = make_llm()
    db = build_or_load_db(researchpapers_dir,CHROMA_DIR,)
    page_documents = extract_pdf_documents(researchpapers_dir)
    chunks = chunk_documents(page_documents)
    run_evaluation(chunks,db)
    validate_framework(chunks,db)

# %% [markdown]
# ## Step 6 — Your written responses in the Capstone Checkpoint 3.1 worksheet
#
# Complete the Capstone Checkpoint 3.1 worksheet using evidence from your capstone evaluation.
# Address the seven sections: system overview, evaluation design, testing approach, baseline results, 
# performance analysis, evaluation framework validation, and reflection and next steps.
#
# 1. **System overview** — your scenario and your 2.1 baseline retriever.
# 2. **Evaluation design** — your criteria and metric (what "correct" means; how the
#    judge decides pass/fail; any thresholds).
# 3. **Testing approach** — how you built your evaluation set and ran it.
# 4. **Baseline results** — the pass/fail outcomes and where the system falls short.
# 5. **Performance analysis** — what the results reveal about strengths, weaknesses,
#    and failure modes.
# 6. **Evaluation framework validation** — show your framework detects a degraded or
#    manipulated output (use the Step 5 result, or your own).
# 7. **Reflection and next steps** — limitations of your evaluation and what you'll
#    improve (this motivates the advanced retrieval in Checkpoint 4.1).
