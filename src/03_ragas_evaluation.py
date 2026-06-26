"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
SYSTEM_V1 = (
    "Bạn là trợ lý AI hữu ích và thân thiện. Chỉ dùng context được cung cấp để trả lời. "
    "Giữ câu trả lời ngắn gọn trong 2-4 câu, rõ ý và dễ hiểu. "
    "Nếu context không có thông tin phù hợp, hãy nói rằng bạn không tìm thấy thông tin.\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Đọc kỹ context, xác định các facts liên quan, "
    "rồi trả lời có cấu trúc trong 3-5 câu. Luôn bám sát dữ liệu được cung cấp, "
    "không suy đoán ngoài context, và nêu rõ khi thông tin chưa đủ.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
RAGAS_SAMPLE_SIZE = int(os.getenv("RAGAS_SAMPLE_SIZE", "50"))
RAGAS_TIMEOUT = int(os.getenv("RAGAS_TIMEOUT", "240"))
RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "8"))
RAGAS_BATCH_SIZE = int(os.getenv("RAGAS_BATCH_SIZE", "8"))
RAGAS_MAX_RETRIES = int(os.getenv("RAGAS_MAX_RETRIES", "1"))
RAGAS_ANSWER_RELEVANCY_STRICTNESS = int(os.getenv("RAGAS_ANSWER_RELEVANCY_STRICTNESS", "1"))
REFRESH_RAG_CACHE = os.getenv("REFRESH_RAG_CACHE", "false").lower() == "true"
CACHE_DIR = Path(__file__).parent.parent / "data" / "ragas_cache"


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    contexts = [doc.page_content for doc in docs]

    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def get_eval_pairs() -> list:
    """Mặc định chạy đủ 50 câu theo yêu cầu lab; env var chỉ dùng khi test nhanh."""
    if RAGAS_SAMPLE_SIZE <= 0 or RAGAS_SAMPLE_SIZE >= len(QA_PAIRS):
        return QA_PAIRS
    return QA_PAIRS[:RAGAS_SAMPLE_SIZE]


def get_cache_path(prompt_version: str, sample_size: int) -> Path:
    return CACHE_DIR / f"{prompt_version}_{sample_size}_rag_outputs.json"


def load_cached_outputs(prompt_version: str, qa_pairs: list) -> list | None:
    if REFRESH_RAG_CACHE:
        return None

    cache_path = get_cache_path(prompt_version, len(qa_pairs))
    if not cache_path.exists():
        return None

    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"⚠️  Cache lỗi JSON, sẽ chạy lại: {cache_path}")
        return None

    if len(cached) != len(qa_pairs):
        print(f"⚠️  Cache không khớp số câu, sẽ chạy lại: {cache_path}")
        return None

    print(f"♻️  Dùng cache RAG outputs cho prompt {prompt_version}: {cache_path}")
    return cached


def save_cached_outputs(prompt_version: str, results: list) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(prompt_version, len(results))
    cache_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"💾 Đã cache RAG outputs: {cache_path}")


def collect_rag_outputs(vectorstore, prompt_version: str, qa_pairs: list) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    cached = load_cached_outputs(prompt_version, qa_pairs)
    if cached is not None:
        return cached

    results = []
    total = len(qa_pairs)
    print(f"\n🚀 Đang chạy {total} câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(qa_pairs, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/{total}] {qa['question'][:60]}")

    save_cached_outputs(prompt_version, results)
    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


def get_ragas_metrics() -> list:
    answer_relevancy.strictness = RAGAS_ANSWER_RELEVANCY_STRICTNESS
    return [faithfulness, answer_relevancy, context_recall, context_precision]


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(
        f"\n📐 Đang đánh giá RAGAS cho prompt {version} "
        f"({len(rag_results)} câu, timeout={RAGAS_TIMEOUT}s, workers={RAGAS_MAX_WORKERS}, "
        f"batch={RAGAS_BATCH_SIZE}, retries={RAGAS_MAX_RETRIES})"
    )

    dataset = build_ragas_dataset(rag_results)
    metrics = get_ragas_metrics()

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=RunConfig(
            timeout=RAGAS_TIMEOUT,
            max_retries=RAGAS_MAX_RETRIES,
            max_wait=15,
            max_workers=RAGAS_MAX_WORKERS,
        ),
        batch_size=RAGAS_BATCH_SIZE,
    )

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        values = [v for v in raw if v is not None]
        scores[key] = float(np.mean(values)) if values else 0.0

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()
    qa_pairs = get_eval_pairs()
    print(
        f"ℹ️  RAGAS sẽ đánh giá {len(qa_pairs)}/{len(QA_PAIRS)} câu hỏi. "
        "Mặc định là đủ 50 câu theo yêu cầu lab."
    )

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(vectorstore, "v1", qa_pairs)
    v2_results = collect_rag_outputs(vectorstore, "v2", qa_pairs)

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "sample_size": len(qa_pairs),
        "total_qa_pairs": len(QA_PAIRS),
        "ragas_config": {
            "timeout": RAGAS_TIMEOUT,
            "max_workers": RAGAS_MAX_WORKERS,
            "batch_size": RAGAS_BATCH_SIZE,
            "max_retries": RAGAS_MAX_RETRIES,
            "answer_relevancy_strictness": RAGAS_ANSWER_RELEVANCY_STRICTNESS,
        },
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
