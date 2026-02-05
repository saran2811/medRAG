import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from typing import Tuple, Dict, List
import time



BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LORA_PATH = "./models"
DB_PATH = "./vectordb"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


RERANK_THRESHOLD = 0.3  
MAX_CONTEXT_CHARS = 1500



print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading LLM (CPU)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu",
    torch_dtype=torch.float32
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

print("Loading embeddings...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'}
)

print("Loading vector database...")
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding
)

print("Loading reranker...")
reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

print("✓ All models loaded!")




def detect_language(text: str) -> str:
    """Detect Chinese vs English"""
    for c in text:
        if '\u4e00' <= c <= '\u9fff':
            return "zh"
    return "en"


def clean_answer(raw: str) -> str:
    """Clean model output"""
    answer = raw
    
    # Extract answer part
    for marker in ["Final Answer:", "Answer:", "回答:", "回答："]:
        if marker in answer:
            answer = answer.split(marker)[-1]
    
    # Remove artifacts
    for junk in ["</think>", "<think>", "\n\n"]:
        answer = answer.replace(junk, " ")
    
    answer = answer.strip()
    
    # Keep first 2 sentences
    sentences = answer.replace("。", ".").split(".")
    sentences = [s.strip() for s in sentences if s.strip()]
    answer = ". ".join(sentences[:2])
    
    if answer and not answer.endswith("."):
        answer += "."
    
    return answer



MEDICAL_TERMS = {
    'diagnosis', 'treatment', 'symptom', 'syndrome', 'prognosis',
    'differential', 'chronic', 'acute', 'therapy', 'contraindication',
    'medication', 'pathology', 'etiology', 'mechanism', 'pharmacology'
}


def adaptive_k(question: str) -> Tuple[int, str]:
    
    words = len(question.split())
    term_count = sum(1 for t in MEDICAL_TERMS if t in question.lower())
    
    if words < 8 and term_count < 1:
        return 3, "simple"
    elif words < 20 and term_count < 3:
        return 5, "moderate"
    else:
        return 7, "complex"


def rerank_documents(question: str, docs: list) -> List[Tuple[any, float]]:
   
    if not docs:
        return []
    

    pairs = [(question, doc.page_content[:500]) for doc in docs]
    

    scores = reranker.predict(pairs)
    
   
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return scored_docs


def corrective_filter(scored_docs: List[Tuple[any, float]], threshold: float = RERANK_THRESHOLD) -> List[Tuple[any, float]]:

    filtered = [(doc, score) for doc, score in scored_docs if score >= threshold]
    
    # Ensure at least 1 document
    if not filtered and scored_docs:
        filtered = [scored_docs[0]]  # Keep top doc
    
    return filtered



#  MAIN RAG FUNCTION


def rag_answer(question: str, use_hybrid: bool = True) -> Tuple[str, str, Dict]:
 
    start_time = time.time()
    timings = {}
    
    lang = detect_language(question)
    metadata = {
        "language": lang,
        "strategies": []
    }
    

    #  ADAPTIVE RAG

    t0 = time.time()
    
    if use_hybrid:
        k, complexity = adaptive_k(question)
        metadata["strategies"].append("adaptive")
        metadata["complexity"] = complexity
    else:
        k = 3
        complexity = "default"
    
    metadata["k"] = k
    timings["adaptive_ms"] = round((time.time() - t0) * 1000, 2)
    
 
    #  RETRIEVE

    t0 = time.time()
    
    try:
        docs = db.similarity_search(
            query=question,
            k=k,
            filter={"language": lang}
        )
    except:
        # Fallback without language filter
        docs = db.similarity_search(query=question, k=k)
    
    metadata["docs_retrieved"] = len(docs)
    timings["retrieval_ms"] = round((time.time() - t0) * 1000, 2)
    
   
    #  RERANKER
   
    t0 = time.time()
    
    if use_hybrid and docs:
        scored_docs = rerank_documents(question, docs)
        metadata["strategies"].append("reranker")
        
        # Store scores for debugging
        metadata["rerank_scores"] = [
            {"rank": i+1, "score": round(float(score), 3)}
            for i, (doc, score) in enumerate(scored_docs)
        ]
    else:
        scored_docs = [(doc, 0.5) for doc in docs]
    
    timings["rerank_ms"] = round((time.time() - t0) * 1000, 2)
    
    # CORRECTIVE RAG
    t0 = time.time()
    
    if use_hybrid and scored_docs:
        filtered_docs = corrective_filter(scored_docs)
        metadata["strategies"].append("corrective")
        metadata["docs_after_filter"] = len(filtered_docs)
        metadata["threshold"] = RERANK_THRESHOLD
        
        # Get just the docs
        final_docs = [doc for doc, score in filtered_docs]
    else:
        final_docs = docs
    
    timings["corrective_ms"] = round((time.time() - t0) * 1000, 2)
    
  
  
    context_parts = []
    total_chars = 0
    
    for doc in final_docs:
        content = doc.page_content[:500]
        if total_chars + len(content) <= MAX_CONTEXT_CHARS:
            context_parts.append(content)
            total_chars += len(content)
        else:
            break
    
    context = "\n---\n".join(context_parts)
    metadata["context_docs"] = len(context_parts)
    
   #ANSWER GENERATION
    t0 = time.time()
    
    prompt = f"""You are a medical expert.Follow the instructions and Important instructions below carefully.
Instructions:
Answer the medical question below in 1-4 sentences using the context.
Be concise and accurate. Do NOT repeat the context.

***Important instructions: 
1) Answer in the language of the question Using the context only.Answer should be complete and concise like a medical expert.
2) Answer should not have repetitive content and should be a whole, coherent answer.***
***Example: Donot repeat like this "Hypertension is when blood pressure is 130 mmHg or higher. Hypertension is when blood pressure is 130 mmHg or higher."***

Context:
{context}

Question: {question}

Final Answer:"""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    raw_output = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = clean_answer(raw_output)
    
    timings["generation_ms"] = round((time.time() - t0) * 1000, 2)
   
    total_ms = round((time.time() - start_time) * 1000, 2)
    
    metadata["timings"] = timings
    metadata["total_ms"] = total_ms
    metadata["retrieval_total_ms"] = (
        timings["adaptive_ms"] + 
        timings["retrieval_ms"] + 
        timings["rerank_ms"] + 
        timings["corrective_ms"]
    )
    print(f"answer: {answer}")
    return answer, context

