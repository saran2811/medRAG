from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Generator
import time
import re
import math
import json
from collections import Counter


from rag_pipeline import rag_answer

app = FastAPI(title="Medical RAG System")


# HELPER FUNCTIONS


def to_float(value) -> float:
    """Convert any numeric type to Python float"""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def sanitize_for_json(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)


# OFFLINE EVALUATION METRICS


class OfflineEvaluator:

    
    def __init__(self):
        self.sentence_model = None
        self._load_sentence_model()
    
    def _load_sentence_model(self):
      
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Sentence transformer loaded for semantic evaluation")
        except ImportError:
            print("⚠ sentence-transformers not installed. Using lexical metrics only.")
            self.sentence_model = None
        except Exception as e:
            print(f"⚠ Failed to load sentence transformer: {e}")
            self.sentence_model = None
    
    def tokenize(self, text: str) -> List[str]:
     
        if not text:
            return []
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text.lower())
        tokens = text.split()
        result = []
        for token in tokens:
            if re.match(r'^[\u4e00-\u9fff]+$', token):
                result.extend(list(token))
            else:
                result.append(token)
        return [t for t in result if t.strip()]
    
    def get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
      
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def word_overlap_f1(self, text1: str, text2: str) -> float:
       
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        overlap = tokens1 & tokens2
        precision = len(overlap) / len(tokens1) if tokens1 else 0
        recall = len(overlap) / len(tokens2) if tokens2 else 0
        
        if precision + recall == 0:
            return 0.0
        return to_float(2 * (precision * recall) / (precision + recall))
    
    def rouge_l(self, reference: str, candidate: str) -> float:

        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        m, n = len(ref_tokens), len(cand_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == cand_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        precision = lcs_length / n if n > 0 else 0
        recall = lcs_length / m if m > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        return to_float(2 * precision * recall / (precision + recall))
    
    def bleu_score(self, reference: str, candidate: str, max_n: int = 4) -> float:

        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)
        
        if not cand_tokens or not ref_tokens:
            return 0.0
        
        precisions = []
        for n in range(1, min(max_n + 1, len(cand_tokens) + 1)):
            ref_ngrams = Counter(self.get_ngrams(ref_tokens, n))
            cand_ngrams = Counter(self.get_ngrams(cand_tokens, n))
            
            if not cand_ngrams:
                continue
                
            overlap = sum((cand_ngrams & ref_ngrams).values())
            total = sum(cand_ngrams.values())
            
            if total == 0:
                precisions.append(0)
            else:
                precisions.append(overlap / total)
        
        if not precisions or all(p == 0 for p in precisions):
            return 0.0
        
        smoothed_precisions = [max(p, 1e-10) for p in precisions]
        log_precisions = [math.log(p) for p in smoothed_precisions]
        avg_log_precision = sum(log_precisions) / len(log_precisions)
        
        bp = 1.0 if len(cand_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
        
        return to_float(bp * math.exp(avg_log_precision))
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
  
        if not text1 or not text2:
            return 0.0
            
        if self.sentence_model is None:
            return self.word_overlap_f1(text1, text2)
        
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            emb1 = [float(x) for x in embeddings[0]]
            emb2 = [float(x) for x in embeddings[1]]
            
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = math.sqrt(sum(a * a for a in emb1))
            norm2 = math.sqrt(sum(b * b for b in emb2))
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = dot_product / (norm1 * norm2)
            return to_float((cosine_sim + 1) / 2)
        except Exception as e:
            return self.word_overlap_f1(text1, text2)
    
    def faithfulness(self, answer: str, context: str) -> float:
  
        if not answer or not context:
            return 0.0
        
        answer_tokens = set(self.tokenize(answer))
        context_tokens = set(self.tokenize(context))
        
        if not answer_tokens:
            return 0.0
        
        grounded = len(answer_tokens & context_tokens) / len(answer_tokens)
        semantic = self.semantic_similarity(answer, context)
        
        return to_float(0.5 * grounded + 0.5 * semantic)
    
    def answer_relevancy(self, question: str, answer: str) -> float:
     
        if not question or not answer:
            return 0.0
        return to_float(self.semantic_similarity(question, answer))
    
    def context_precision(self, question: str, context: str) -> float:
   
        if not question or not context:
            return 0.0
        return to_float(self.semantic_similarity(question, context))
    
    def evaluate_single(self, question: str, answer: str, context: str, ground_truth: str) -> Dict[str, float]:
     
        return {
            "faithfulness": round(to_float(self.faithfulness(answer, context)), 4),
            "answer_relevancy": round(to_float(self.answer_relevancy(question, answer)), 4),
            "context_precision": round(to_float(self.context_precision(question, context)), 4),
            "correctness_f1": round(to_float(self.word_overlap_f1(answer, ground_truth)), 4),
            "correctness_rouge_l": round(to_float(self.rouge_l(ground_truth, answer)), 4),
            "correctness_bleu": round(to_float(self.bleu_score(ground_truth, answer)), 4),
            "correctness_semantic": round(to_float(self.semantic_similarity(answer, ground_truth)), 4)
        }



evaluator = OfflineEvaluator()



#  MODELS


class Query(BaseModel):
    question: str


class EvalQuestion(BaseModel):
    question: str
    ground_truth: str


class EvalRequest(BaseModel):
    questions: List[EvalQuestion]



#   EVALUATION DATA


DEFAULT_EVAL_DATA = [
    {
        "question": "What is hypertension?",
        "ground_truth": "Hypertension is chronic high blood pressure, typically BP ≥140/90 mmHg."
    },
    {
        "question": "What are symptoms of diabetes?",
        "ground_truth": "Symptoms include increased thirst, frequent urination, fatigue, and blurred vision."
    },
    {
        "question": "What is the first-line treatment for Type 2 Diabetes?",
        "ground_truth": "Metformin is the first-line treatment along with lifestyle modifications."
    },
    {
        "question": "What are contraindications for NSAIDs?",
        "ground_truth": "Contraindications include peptic ulcer, renal impairment, and cardiovascular disease."
    },
    {
        "question": "糖尿病的主要原因是什么？",
        "ground_truth": "糖尿病主要由胰岛素分泌不足或胰岛素抵抗引起。"
    }
]


# ============================================================================
#                              API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home():
  
    return get_html_template()


@app.get("/health")
def health():
    return {"status": "running", "evaluator": "offline"}


@app.post("/ask")
def ask(q: Query):
    
    start = time.time()
    
    try:
        answer, context = rag_answer(q.question)
        elapsed = round((time.time() - start) * 1000, 2)
        
        return {
            "success": True,
            "question": q.question,
            "answer": str(answer),
            "context": str(context),
            "time_ms": float(elapsed)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/eval/questions")
def get_eval_questions():
  
    return {"questions": DEFAULT_EVAL_DATA}


@app.get("/eval/stream")
async def stream_evaluation():

    
    def generate() -> Generator[str, None, None]:
        eval_data = DEFAULT_EVAL_DATA
        all_scores = []
        total_time = 0
        
        # Send start event
        yield f"data: {json.dumps({'type': 'start', 'total': len(eval_data)})}\n\n"
        
        for idx, item in enumerate(eval_data):
            start = time.time()
            
            try:
                # Get RAG answer
                answer, context = rag_answer(item["question"])
                elapsed = round((time.time() - start) * 1000, 2)
                total_time += elapsed
                
                # Calculate scores
                scores = evaluator.evaluate_single(
                    question=item["question"],
                    answer=answer,
                    context=context,
                    ground_truth=item["ground_truth"]
                )
                
                # Calculate overall score for this question
                overall = round(to_float(
                    0.2 * scores["faithfulness"] +
                    0.2 * scores["answer_relevancy"] +
                    0.2 * scores["context_precision"] +
                    0.4 * scores["correctness_semantic"]
                ), 4)
                
                result = {
                    "type": "result",
                    "index": idx,
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "answer": answer,
                    "context": context[:500] + "..." if len(context) > 500 else context,
                    "scores": scores,
                    "overall": overall,
                    "time_ms": elapsed,
                    "status": "success"
                }
                
                all_scores.append(scores)
                
            except Exception as e:
                result = {
                    "type": "result",
                    "index": idx,
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "answer": "",
                    "context": "",
                    "error": str(e),
                    "status": "error",
                    "time_ms": 0
                }
            
            yield f"data: {json.dumps(sanitize_for_json(result))}\n\n"
        
  
        if all_scores:
            metrics = ["faithfulness", "answer_relevancy", "context_precision", 
                       "correctness_f1", "correctness_rouge_l", "correctness_bleu", "correctness_semantic"]
            
            aggregated = {}
            for metric in metrics:
                values = [to_float(s[metric]) for s in all_scores]
                aggregated[metric] = round(sum(values) / len(values), 4)
            
            aggregated["overall"] = round(to_float(
                0.2 * aggregated["faithfulness"] +
                0.2 * aggregated["answer_relevancy"] +
                0.2 * aggregated["context_precision"] +
                0.4 * aggregated["correctness_semantic"]
            ), 4)
        else:
            aggregated = {}
        

        summary = {
            "type": "complete",
            "total_questions": len(eval_data),
            "successful": len(all_scores),
            "avg_time_ms": round(total_time / len(eval_data), 2) if eval_data else 0,
            "final_scores": aggregated
        }
        
        yield f"data: {json.dumps(sanitize_for_json(summary))}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )



# HTML TEMPLATE


def get_html_template():
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical RAG System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Header */
        .header {
            text-align: center;
            padding: 40px 0;
        }
        
        .header h1 {
            font-size: 2.8rem;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header p {
            opacity: 0.8;
            font-size: 1.1rem;
        }
        
        .badge {
            display: inline-block;
            padding: 8px 20px;
            background: rgba(102, 126, 234, 0.2);
            border: 1px solid rgba(102, 126, 234, 0.5);
            border-radius: 25px;
            font-size: 0.85rem;
            margin-top: 15px;
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
            justify-content: center;
        }
        
        .tab-btn {
            padding: 14px 35px;
            border: none;
            background: rgba(255,255,255,0.1);
            color: white;
            border-radius: 12px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .tab-btn:hover {
            background: rgba(255,255,255,0.15);
            transform: translateY(-2px);
        }
        
        .tab-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-color: transparent;
            font-weight: 600;
        }
        
        /* Cards */
        .card {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 20px;
        }
        
        .card h2 {
            color: #fff;
            margin-bottom: 20px;
            font-size: 1.5rem;
        }
        
        /* Input Section */
        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .input-group input {
            flex: 1;
            padding: 16px 22px;
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            font-size: 1rem;
            background: rgba(255,255,255,0.05);
            color: white;
            transition: all 0.3s;
        }
        
        .input-group input::placeholder {
            color: rgba(255,255,255,0.5);
        }
        
        .input-group input:focus {
            outline: none;
            border-color: #667eea;
            background: rgba(255,255,255,0.1);
        }
        
        .btn {
            padding: 16px 35px;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        
        .btn-success {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }
        
        .btn-success:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(56, 239, 125, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }
        
        /* Answer Box */
        .answer-box {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 25px;
            margin-top: 20px;
            display: none;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .answer-box.show {
            display: block;
            animation: slideUp 0.5s ease;
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .answer-label {
            font-size: 0.8rem;
            color: rgba(255,255,255,0.6);
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 10px;
        }
        
        .answer-text {
            font-size: 1.15rem;
            color: #fff;
            line-height: 1.7;
        }
        
        .context-box {
            margin-top: 20px;
            padding: 20px;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 12px;
            border-left: 4px solid #667eea;
        }
        
        .context-text {
            font-size: 0.9rem;
            color: rgba(255,255,255,0.8);
            line-height: 1.6;
            max-height: 150px;
            overflow-y: auto;
        }
        
        .time-badge {
            display: inline-block;
            padding: 8px 16px;
            background: rgba(56, 239, 125, 0.2);
            color: #38ef7d;
            border-radius: 25px;
            font-size: 0.85rem;
            margin-top: 15px;
        }
        
        /* Sample Questions */
        .sample-questions {
            margin-top: 20px;
        }
        
        .sample-questions p {
            font-size: 0.9rem;
            color: rgba(255,255,255,0.6);
            margin-bottom: 12px;
        }
        
        .sample-btn {
            display: inline-block;
            padding: 10px 18px;
            margin: 5px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 25px;
            cursor: pointer;
            font-size: 0.85rem;
            color: white;
            transition: all 0.2s;
        }
        
        .sample-btn:hover {
            background: rgba(102, 126, 234, 0.3);
            border-color: #667eea;
            transform: translateY(-2px);
        }
        
        /* Evaluation Section */
        .eval-section {
            display: none;
        }
        
        .eval-section.show {
            display: block;
        }
        
        /* Progress Section */
        .progress-section {
            margin: 30px 0;
        }
        
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .progress-title {
            font-size: 1.1rem;
            color: rgba(255,255,255,0.9);
        }
        
        .progress-count {
            font-size: 0.9rem;
            color: rgba(255,255,255,0.6);
        }
        
        .progress-bar-container {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            height: 12px;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            transition: width 0.5s ease;
            width: 0%;
        }
        
        /* Question Cards */
        .question-cards {
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-top: 30px;
        }
        
        .question-card {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.1);
            overflow: hidden;
            transition: all 0.3s;
        }
        
        .question-card.processing {
            border-color: #667eea;
            box-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
        }
        
        .question-card.success {
            border-color: rgba(56, 239, 125, 0.5);
        }
        
        .question-card.error {
            border-color: rgba(255, 107, 107, 0.5);
        }
        
        .question-card-header {
            padding: 20px 25px;
            background: rgba(255,255,255,0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .question-number {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
        }
        
        .question-text {
            flex: 1;
            margin-left: 15px;
            font-size: 1.05rem;
            color: #fff;
        }
        
        .question-status {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .status-pending {
            background: rgba(255,255,255,0.1);
            color: rgba(255,255,255,0.5);
        }
        
        .status-processing {
            background: rgba(102, 126, 234, 0.2);
            color: #667eea;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .status-success {
            background: rgba(56, 239, 125, 0.2);
            color: #38ef7d;
        }
        
        .status-error {
            background: rgba(255, 107, 107, 0.2);
            color: #ff6b6b;
        }
        
        .question-card-body {
            padding: 25px;
            display: none;
        }
        
        .question-card.success .question-card-body,
        .question-card.error .question-card-body {
            display: block;
            animation: slideUp 0.4s ease;
        }
        
        .answer-section {
            margin-bottom: 20px;
        }
        
        .answer-section h4 {
            font-size: 0.8rem;
            color: rgba(255,255,255,0.5);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        
        .answer-content {
            background: rgba(255,255,255,0.05);
            padding: 15px 20px;
            border-radius: 10px;
            font-size: 0.95rem;
            line-height: 1.6;
            color: rgba(255,255,255,0.9);
        }
        
        .ground-truth {
            background: rgba(56, 239, 125, 0.1);
            border-left: 3px solid #38ef7d;
        }
        
        .generated-answer {
            background: rgba(102, 126, 234, 0.1);
            border-left: 3px solid #667eea;
        }
        
        /* Scores Grid in Question Card */
        .scores-mini-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 20px;
        }
        
        .score-mini-card {
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .score-mini-value {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .score-mini-value.high { color: #38ef7d; }
        .score-mini-value.medium { color: #f6d365; }
        .score-mini-value.low { color: #ff6b6b; }
        
        .score-mini-label {
            font-size: 0.7rem;
            color: rgba(255,255,255,0.5);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .question-time {
            text-align: right;
            margin-top: 15px;
            font-size: 0.85rem;
            color: rgba(255,255,255,0.4);
        }
        
        /* Final Results */
        .final-results {
            display: none;
            margin-top: 40px;
            animation: slideUp 0.5s ease;
        }
        
        .final-results.show {
            display: block;
        }
        
        .final-header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .final-header h2 {
            font-size: 2rem;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #38ef7d, #11998e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .overall-score-display {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
            border: 2px solid rgba(102, 126, 234, 0.5);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
        }
        
        .overall-score-value {
            font-size: 5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .overall-score-label {
            font-size: 1.2rem;
            color: rgba(255,255,255,0.7);
            margin-top: 10px;
        }
        
        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .metric-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .metric-name {
            font-size: 1rem;
            font-weight: 600;
            color: #fff;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 800;
        }
        
        .metric-value.excellent { color: #38ef7d; }
        .metric-value.good { color: #96e6a1; }
        .metric-value.fair { color: #f6d365; }
        .metric-value.poor { color: #ff6b6b; }
        
        .metric-bar {
            background: rgba(255,255,255,0.1);
            height: 8px;
            border-radius: 4px;
            margin: 15px 0;
            overflow: hidden;
        }
        
        .metric-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 1s ease;
        }
        
        .metric-bar-fill.excellent { background: linear-gradient(90deg, #11998e, #38ef7d); }
        .metric-bar-fill.good { background: linear-gradient(90deg, #96e6a1, #d4fc79); }
        .metric-bar-fill.fair { background: linear-gradient(90deg, #f6d365, #fda085); }
        .metric-bar-fill.poor { background: linear-gradient(90deg, #ff6b6b, #ff9a9e); }
        
        .metric-description {
            font-size: 0.85rem;
            color: rgba(255,255,255,0.5);
            line-height: 1.5;
        }
        
        /* Summary Stats */
        .summary-stats {
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 30px;
        }
        
        .summary-stat {
            background: rgba(255,255,255,0.05);
            padding: 20px 30px;
            border-radius: 12px;
            text-align: center;
            min-width: 150px;
        }
        
        .summary-stat-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #fff;
        }
        
        .summary-stat-label {
            font-size: 0.85rem;
            color: rgba(255,255,255,0.5);
            margin-top: 5px;
        }
        
        /* Loading Spinner */
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(255,255,255,0.1);
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .header h1 { font-size: 2rem; }
            .input-group { flex-direction: column; }
            .tabs { flex-direction: column; }
            .overall-score-value { font-size: 3.5rem; }
            .metrics-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🏥 Medical RAG System</h1>
            <p>AI-powered medical question answering with hybrid RAG</p>
            <span class="badge">🔒 Offline Evaluation </span>
        </div>
        
        <!-- Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="showTab('qa')">💬 Ask Question</button>
            <button class="tab-btn" onclick="showTab('eval')">📊 Run Evaluation</button>
        </div>
        
        <!-- Q&A Section -->
        <div id="qa-section" class="card">
            <h2>Ask a Medical Question</h2>
            
            <div class="input-group">
                <input type="text" id="question-input" placeholder="Enter your medical question..." 
                       onkeypress="if(event.key==='Enter') askQuestion()">
                <button class="btn btn-primary" onclick="askQuestion()" id="ask-btn">Ask</button>
            </div>
            
            <div class="sample-questions">
                <p>✨ Try these examples:</p>
                <button class="sample-btn" onclick="setQuestion('Explain hypertension')">Explain hypertension</button>
                <button class="sample-btn" onclick="setQuestion('What are symptoms of diabetes?')">Symptoms of diabetes</button>
                <button class="sample-btn" onclick="setQuestion('What is the treatment for pneumonia?')">Pneumonia treatment</button>
                <button class="sample-btn" onclick="setQuestion('糖尿病的症状是什么？')">糖尿病症状 (Chinese)</button>
            </div>
            
            <div class="loading" id="qa-loading">
                <div class="spinner"></div>
                <p style="color: rgba(255,255,255,0.7);">Generating answer...</p>
            </div>
            
            <div class="answer-box" id="answer-box">
                <div class="answer-label">Answer</div>
                <div class="answer-text" id="answer-text"></div>
                
                <div class="context-box">
                    <div class="answer-label">Retrieved Context</div>
                    <div class="context-text" id="context-text"></div>
                </div>
                
                <span class="time-badge" id="time-badge"></span>
            </div>
        </div>
        
        <!-- Evaluation Section -->
        <div id="eval-section" class="card eval-section">
            <h2>📊 Interactive Evaluation Suite</h2>
            <p style="color: rgba(255,255,255,0.6); margin-bottom: 25px;">
                Watch as each question is processed in real-time. Get detailed metrics and explanations for your RAG system's performance.
            </p>
            
            <button class="btn btn-success" onclick="runStreamingEvaluation()" id="eval-btn">
                🚀 Start Evaluation
            </button>
            
            <!-- Progress Section -->
            <div class="progress-section" id="progress-section" style="display: none;">
                <div class="progress-header">
                    <span class="progress-title">⏳ Processing Questions...</span>
                    <span class="progress-count" id="progress-count">0 / 0</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progress-bar"></div>
                </div>
            </div>
            
            <!-- Question Cards -->
            <div class="question-cards" id="question-cards"></div>
            
            <!-- Final Results -->
            <div class="final-results" id="final-results">
                <div class="final-header">
                    <h2>✅ Evaluation Complete!</h2>
                    <p style="color: rgba(255,255,255,0.6);">Here's how your RAG system performed</p>
                </div>
                
                <!-- Overall Score -->
                <div class="overall-score-display">
                    <div class="overall-score-value" id="final-overall-score">--%</div>
                    <div class="overall-score-label">Overall Score</div>
                </div>
                
                <!-- Metrics Grid -->
                <h3 style="margin-bottom: 20px; color: rgba(255,255,255,0.9);">📈 Detailed Metrics</h3>
                <div class="metrics-grid" id="metrics-grid"></div>
                
                <!-- Summary Stats -->
                <div class="summary-stats" id="summary-stats"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Metric explanations
        const metricExplanations = {
            faithfulness: {
                name: "Faithfulness",
                icon: "🎯",
                description: "Measures if the generated answer is grounded in the retrieved context. High score means the answer doesn't hallucinate."
            },
            answer_relevancy: {
                name: "Answer Relevancy",
                icon: "💡",
                description: "Evaluates how well the answer addresses the question asked. High score means the answer is on-topic."
            },
            context_precision: {
                name: "Context Precision",
                icon: "🔍",
                description: "Measures if the retrieved context is relevant to the question. High score means good retrieval quality."
            },
            correctness_f1: {
                name: "F1 Score",
                icon: "📊",
                description: "Word overlap between generated answer and ground truth. Balances precision and recall."
            },
            correctness_rouge_l: {
                name: "ROUGE-L",
                icon: "📏",
                description: "Longest common subsequence between answer and ground truth. Captures sentence-level structure."
            },
            correctness_bleu: {
                name: "BLEU Score",
                icon: "📝",
                description: "N-gram precision score commonly used in translation. Measures phrase-level accuracy."
            },
            correctness_semantic: {
                name: "Semantic Similarity",
                icon: "🧠",
                description: "Deep learning-based similarity between answer and ground truth. Captures meaning beyond exact words."
            }
        };
        
        // Tab switching
        function showTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('qa-section').style.display = tab === 'qa' ? 'block' : 'none';
            document.getElementById('eval-section').classList.toggle('show', tab === 'eval');
        }
        
        // Set sample question
        function setQuestion(q) {
            document.getElementById('question-input').value = q;
        }
        
        // Ask question
        async function askQuestion() {
            const input = document.getElementById('question-input');
            const question = input.value.trim();
            
            if (!question) { alert('Please enter a question'); return; }
            
            const btn = document.getElementById('ask-btn');
            const loading = document.getElementById('qa-loading');
            const answerBox = document.getElementById('answer-box');
            
            btn.disabled = true;
            loading.classList.add('show');
            answerBox.classList.remove('show');
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('answer-text').textContent = data.answer;
                    document.getElementById('context-text').textContent = data.context;
                    document.getElementById('time-badge').textContent = `⏱️ ${data.time_ms} ms`;
                    answerBox.classList.add('show');
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                btn.disabled = false;
                loading.classList.remove('show');
            }
        }
        
        // Get score class
        function getScoreClass(score) {
            if (score >= 0.85) return 'excellent';
            if (score >= 0.70) return 'good';
            if (score >= 0.50) return 'fair';
            return 'poor';
        }
        
        function getMiniScoreClass(score) {
            if (score >= 0.70) return 'high';
            if (score >= 0.50) return 'medium';
            return 'low';
        }
        
        // Run streaming evaluation
        async function runStreamingEvaluation() {
            const btn = document.getElementById('eval-btn');
            const progressSection = document.getElementById('progress-section');
            const questionCards = document.getElementById('question-cards');
            const finalResults = document.getElementById('final-results');
            
            btn.disabled = true;
            btn.textContent = '⏳ Running...';
            progressSection.style.display = 'block';
            questionCards.innerHTML = '';
            finalResults.classList.remove('show');
            
            let totalQuestions = 0;
            let processedQuestions = 0;
            
            try {
                const eventSource = new EventSource('/eval/stream');
                
                eventSource.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'start') {
                        totalQuestions = data.total;
                        // Create pending cards for all questions
                        for (let i = 0; i < totalQuestions; i++) {
                            questionCards.innerHTML += createPendingCard(i);
                        }
                        updateProgress(0, totalQuestions);
                    }
                    else if (data.type === 'result') {
                        processedQuestions++;
                        updateProgress(processedQuestions, totalQuestions);
                        updateQuestionCard(data);
                    }
                    else if (data.type === 'complete') {
                        eventSource.close();
                        displayFinalResults(data);
                        btn.disabled = false;
                        btn.textContent = '🚀 Start Evaluation';
                    }
                };
                
                eventSource.onerror = function(error) {
                    console.error('EventSource error:', error);
                    eventSource.close();
                    btn.disabled = false;
                    btn.textContent = '🚀 Start Evaluation';
                    alert('Evaluation stream error. Please try again.');
                };
                
            } catch (error) {
                alert('Error: ' + error.message);
                btn.disabled = false;
                btn.textContent = '🚀 Start Evaluation';
            }
        }
        
        // Create pending card
        function createPendingCard(index) {
            return `
                <div class="question-card" id="question-card-${index}">
                    <div class="question-card-header">
                        <div style="display: flex; align-items: center;">
                            <div class="question-number">${index + 1}</div>
                            <div class="question-text" id="question-text-${index}">Waiting...</div>
                        </div>
                        <span class="question-status status-pending" id="question-status-${index}">Pending</span>
                    </div>
                    <div class="question-card-body" id="question-body-${index}"></div>
                </div>
            `;
        }
        
        // Update progress
        function updateProgress(current, total) {
            const percentage = (current / total) * 100;
            document.getElementById('progress-bar').style.width = percentage + '%';
            document.getElementById('progress-count').textContent = `${current} / ${total}`;
            
            // Update next card to processing
            if (current < total) {
                const nextCard = document.getElementById(`question-card-${current}`);
                const nextStatus = document.getElementById(`question-status-${current}`);
                if (nextCard && nextStatus) {
                    nextCard.classList.add('processing');
                    nextStatus.className = 'question-status status-processing';
                    nextStatus.textContent = 'Processing...';
                }
            }
        }
        
        // Update question card with result
        function updateQuestionCard(data) {
            const card = document.getElementById(`question-card-${data.index}`);
            const status = document.getElementById(`question-status-${data.index}`);
            const questionText = document.getElementById(`question-text-${data.index}`);
            const body = document.getElementById(`question-body-${data.index}`);
            
            if (!card) return;
            
            card.classList.remove('processing');
            questionText.textContent = data.question;
            
            if (data.status === 'success') {
                card.classList.add('success');
                status.className = 'question-status status-success';
                status.textContent = '✓ Complete';
                
                // Build scores HTML
                const scores = data.scores || {};
                const scoresHtml = Object.entries(scores).slice(0, 4).map(([key, value]) => {
                    const label = key.replace('correctness_', '').replace('_', '-').toUpperCase();
                    const scoreClass = getMiniScoreClass(value);
                    return `
                        <div class="score-mini-card">
                            <div class="score-mini-value ${scoreClass}">${(value * 100).toFixed(0)}%</div>
                            <div class="score-mini-label">${label}</div>
                        </div>
                    `;
                }).join('');
                
                body.innerHTML = `
                    <div class="answer-section">
                        <h4>📋 Ground Truth</h4>
                        <div class="answer-content ground-truth">${data.ground_truth}</div>
                    </div>
                    <div class="answer-section">
                        <h4>🤖 Generated Answer</h4>
                        <div class="answer-content generated-answer">${data.answer}</div>
                    </div>
                    <div class="scores-mini-grid">${scoresHtml}</div>
                    <div class="question-time">⏱️ ${data.time_ms} ms • Overall: ${(data.overall * 100).toFixed(1)}%</div>
                `;
            } else {
                card.classList.add('error');
                status.className = 'question-status status-error';
                status.textContent = '✗ Error';
                
                body.innerHTML = `
                    <div class="answer-section">
                        <h4>❌ Error</h4>
                        <div class="answer-content" style="border-left-color: #ff6b6b;">${data.error || 'Unknown error'}</div>
                    </div>
                `;
            }
        }
        
        // Display final results
        function displayFinalResults(data) {
            const finalResults = document.getElementById('final-results');
            const scores = data.final_scores || {};
            
            // Overall score
            const overall = scores.overall || 0;
            document.getElementById('final-overall-score').textContent = (overall * 100).toFixed(1) + '%';
            
            // Metrics grid
            const metricsGrid = document.getElementById('metrics-grid');
            const metrics = [
                'faithfulness', 'answer_relevancy', 'context_precision',
                'correctness_semantic', 'correctness_f1', 'correctness_rouge_l', 'correctness_bleu'
            ];
            
            metricsGrid.innerHTML = metrics.map(key => {
                const value = scores[key] || 0;
                const info = metricExplanations[key] || { name: key, icon: '📊', description: '' };
                const scoreClass = getScoreClass(value);
                
                return `
                    <div class="metric-card">
                        <div class="metric-card-header">
                            <span class="metric-name">${info.icon} ${info.name}</span>
                            <span class="metric-value ${scoreClass}">${(value * 100).toFixed(1)}%</span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill ${scoreClass}" style="width: ${value * 100}%"></div>
                        </div>
                        <div class="metric-description">${info.description}</div>
                    </div>
                `;
            }).join('');
            
            // Summary stats
            document.getElementById('summary-stats').innerHTML = `
                <div class="summary-stat">
                    <div class="summary-stat-value">${data.successful}/${data.total_questions}</div>
                    <div class="summary-stat-label">Questions Answered</div>
                </div>
                <div class="summary-stat">
                    <div class="summary-stat-value">${data.avg_time_ms}ms</div>
                    <div class="summary-stat-label">Avg Response Time</div>
                </div>
                <div class="summary-stat">
                    <div class="summary-stat-value">${((scores.faithfulness || 0) * 100).toFixed(0)}%</div>
                    <div class="summary-stat-label">Faithfulness</div>
                </div>
                <div class="summary-stat">
                    <div class="summary-stat-value">${((scores.correctness_semantic || 0) * 100).toFixed(0)}%</div>
                    <div class="summary-stat-label">Semantic Match</div>
                </div>
            `;
            
            finalResults.classList.add('show');
            
            // Scroll to results
            finalResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    </script>
</body>
</html>
'''



#   MAIN


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🏥 Medical RAG System - Interactive Evaluation")
    print("=" * 60)
    print("UI:   http://0.0.0.0:8000")
    print("API:  http://0.0.0.0:8000/docs")
    print("=" * 60)
    print("Features:")
    print("  ✓ Real-time streaming evaluation")
    print("  ✓ Step-by-step question processing")
    print("  ✓ Detailed metrics with explanations")
    print("  ✓ No external API required")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
