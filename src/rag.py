# src/rag.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os, json, requests

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# CONFIG
INDEX_DIR  = "embeddings/faiss_index"
#HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
#HF_API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions"
#HEADERS    = {"Authorization": f"Bearer {HF_TOKEN}"}

from groq import Groq

# CONFIG
INDEX_DIR   = "embeddings/faiss_index"
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

# CALL LLM VIA GROQ
def call_mistral(prompt):
    response = GROQ_CLIENT.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.2
    )
    return response.choices[0].message.content

# LOAD VECTOR STORE
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# RETRIEVE RELEVANT CHUNKS
def retrieve_context(vectorstore, ticker, query, year=None, k=5):
    results = vectorstore.similarity_search(query, k=k*3)
    
    # filter by ticker
    filtered = [r for r in results 
                if ticker.upper() in r.metadata.get("source", "").upper()]
    
    # filter by year if provided
    if year:
        year_filtered = [r for r in filtered 
                         if r.metadata.get("year") == year]
        # fall back to ticker-only if no year match
        filtered = year_filtered if year_filtered else filtered

    return filtered[:k]
#

def get_context_from_file(ticker, year):
    import glob
    data_dir = "data/sec-edgar-filings"
    pattern  = f"{data_dir}/{ticker}/10-K/*/full-submission.txt"
    
    for filepath in glob.glob(pattern):
        parts       = filepath.split(os.sep)
        accession   = parts[4] if len(parts) > 4 else ""
        year_digits = accession.split("-")[1] if "-" in accession else "00"
        file_year   = int("20" + year_digits)
        
        if file_year == year:
            with open(filepath, "r", errors="ignore") as f:
                text = f.read()
            
            # find ALL occurrences of ITEM 1A and take the LAST one
            # (first hit is table of contents, last hit is actual content)
            search_terms = ["ITEM\xa01A", "ITEM 1A"]
            best_idx = -1
            
            for term in search_terms:
                idx = 0
                while True:
                    pos = text.upper().find(term.upper(), idx)
                    if pos == -1:
                        break
                    best_idx = pos
                    idx = pos + 1
            
            if best_idx != -1:
                # grab 6000 chars from the last occurrence
                chunk = text[best_idx:best_idx+6000]
                # skip if it's still just a TOC line (too short before next item)
                if len(chunk) > 500:
                    print(f" Found ITEM 1A at position {best_idx} for {ticker} {year}")
                    return chunk
            
            # fallback: search for RISK FACTORS
            idx = text.upper().find("RISK FACTORS")
            if idx != -1:
                chunk = text[idx:idx+6000]
                if len(chunk) > 500:
                    return chunk
            
            print(f"No risk section found for {ticker} {year}, using raw text")
            return text[5000:11000]  # skip header, grab middle section
    
    return None

# GENERATE INVESTMENT SIGNAL
def generate_signal(ticker, query="What are the major risk factors and investment outlook?", year=None):
    print(f"🔍 Retrieving context for {ticker} {f'({year})' if year else ''}...")
    
    # if year specified, read directly from file instead of FAISS
    if year:
        context = get_context_from_file(ticker, year)
        if not context:
            return {"error": f"No filing found for {ticker} {year}"}
    else:
        vectorstore = load_vectorstore()
        chunks = retrieve_context(vectorstore, ticker, query, year=year)
        if not chunks:
            return {"error": f"No filing data found for {ticker}"}
        retrieved_years = list(set([c.metadata.get("year") for c in chunks]))
        print(f"   Retrieved chunks from years: {retrieved_years}")
        context = "\n\n".join([c.page_content for c in chunks])

    prompt = f"""<s>[INST] You are a senior financial analyst reviewing a specific SEC 10-K filing.

Your task: Extract SPECIFIC risks mentioned in THIS filing only. Do not use generic risks like "market volatility" or "regulatory changes" unless they are explicitly discussed with specific details in the text below.

Return ONLY a JSON object with no extra text:
- "ticker": string
- "risk_level": one of "LOW", "MEDIUM", "HIGH"
- "sentiment": one of "BULLISH", "NEUTRAL", "BEARISH"
- "key_risks": list of 3 SPECIFIC risks with details from the filing
- "rationale": one sentence referencing specific numbers or events from the filing
- "red_flags": list of specific warning phrases found in the filing, empty list if none

Filing year: {year if year else 'unknown'}
Company: {ticker}

Filing excerpt:
{context[:4000]}

Respond with ONLY the JSON object. [/INST]"""

    print(f"🤖 Calling Mistral for {ticker}...")
    raw = call_mistral(prompt)

    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"ticker": ticker, "raw_response": raw, "error": "Could not parse JSON"}
    

# TEST
if __name__ == "__main__":
    test_cases = [
        ("AAPL", 2023),
        ("AAPL", 2024),
        ("AAPL", 2025),
        ("JPM",  2024),
        ("JPM",  2025),
        ("JPM",  2026),
    ]
    for ticker, year in test_cases:
        print(f"\n{'='*50} {ticker} ({year})")
        signal = generate_signal(ticker, year=year)
        print(json.dumps(signal, indent=2))