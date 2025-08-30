import os
import json
import numpy as np
import fitz  # PyMuPDF
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASS = os.getenv("MYSQL_PASS", "your_mysql_pass")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_DB   = os.getenv("MYSQL_DB", "career_copilot")
EMB_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Global objects ---
engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}/{MYSQL_DB}")
embedder = SentenceTransformer(EMB_MODEL)

@lru_cache(maxsize=1)
def get_skills_with_embeddings():
    """
    Loads skills and their embeddings from the database.
    Uses lru_cache to ensure this database call runs only once per server start.
    """
    print("--- Loading and caching skills from database... ---")
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT skill_id, name, embedding FROM skills")).mappings().all()
    
    processed_skills = []
    for row in rows:
        skill = dict(row)
        if skill.get('embedding'):
            try:
                skill['emb_array'] = np.array(json.loads(skill['embedding']))
            except (json.JSONDecodeError, TypeError):
                skill['emb_array'] = None
        else:
            skill['emb_array'] = None
        processed_skills.append(skill)
    
    print(f"--- Successfully cached {len(processed_skills)} skills. ---")
    return processed_skills

def parse_pdf_text(pdf_path: str) -> str:
    """Extracts all text from a given PDF file using PyMuPDF (fitz)."""
    try:
        with fitz.open(pdf_path) as doc:
            return "".join(page.get_text() for page in doc).lower()
    except Exception as e:
        print(f"Error parsing PDF at {pdf_path}: {e}")
        return ""

def extract_skills_from_text(text: str, top_k=20, sim_threshold=0.35, chunk_size=100, chunk_overlap=20) -> list:
    """
    Extracts skills by chunking resume text and comparing each chunk's embedding
    against a cached list of skill embeddings.
    """
    skill_rows = get_skills_with_embeddings()
    if not text or not skill_rows:
        return []

    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - chunk_overlap)]
    
    if not chunks:
        return []

    chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True)
    
    skill_similarities = {}

    for skill in skill_rows:
        emb = skill.get('emb_array')
        if emb is None or emb.size == 0:
            continue
        
        # Calculate cosine similarity of this skill against ALL chunk embeddings efficiently
        sims = np.dot(chunk_embeddings, emb) / (np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(emb))
        
        # Find the highest similarity for this skill from any chunk
        max_sim = np.max(sims)
        
        if max_sim >= sim_threshold:
            skill_similarities[skill['name']] = {
                "skill_id": skill['skill_id'],
                "max_sim": float(max_sim)
            }

    sorted_skills = sorted(skill_similarities.items(), key=lambda item: item[1]['max_sim'], reverse=True)
    
    matched = [(item[1]['skill_id'], item[0], item[1]['max_sim']) for item in sorted_skills]
    
    return matched[:top_k]

def save_user_skills(user_id: int, matched_skills: list):
    """Saves a user's matched skills, replacing any old ones."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM user_skills WHERE user_id = :u"), {"u": user_id})
        for sid, name, sim in matched_skills:
            conn.execute(
                text("INSERT INTO user_skills(user_id, skill_id, confidence) VALUES (:u, :s, :c)"),
                {"u": user_id, "s": sid, "c": sim}
            )

def compute_gap(user_id: int, role_pattern: str, top_n=15) -> list:
    """
    Finds top skills for a role that a user is missing, based on aggregated job data.
    """
    q = """
    SELECT s.skill_id, s.name, COUNT(js.job_id) as freq
    FROM skills s
    JOIN job_skills js ON s.skill_id = js.skill_id
    JOIN jobs j ON js.job_id = j.job_id
    WHERE LOWER(j.title) LIKE :pattern
      AND s.skill_id NOT IN (SELECT us.skill_id FROM user_skills us WHERE us.user_id = :user_id)
    GROUP BY s.skill_id, s.name
    ORDER BY freq DESC
    LIMIT :limit;
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text(q),
            {"pattern": f"%{role_pattern.lower()}%", "user_id": user_id, "limit": top_n}
        ).mappings().all()
    # The third value returned is 'freq' (frequency), which represents importance.
    return [(r['skill_id'], r['name'], r['freq']) for r in rows]

