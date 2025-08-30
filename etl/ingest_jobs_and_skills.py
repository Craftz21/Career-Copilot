import os
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASS = os.getenv("MYSQL_PASS", "your_mysql_pass")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_DB = os.getenv("MYSQL_DB", "career_copilot")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Global objects ---
engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}/{MYSQL_DB}")
embedder = SentenceTransformer(EMB_MODEL)

def load_csv_data(path, required_columns):
    """Loads a CSV, validates columns, and cleans NaN values."""
    try:
        df = pd.read_csv(path, header=0, names=required_columns, engine='python', sep=',', quotechar='"')
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] The data file was not found at {path}")
    except pd.errors.ParserError as e:
        raise SystemExit(f"[ERROR] Failed to parse CSV at {path}: {str(e)}")
    df = df.replace({np.nan: None})
    return df

def seed_skills(skills_csv="data/skills_master.csv"):
    """Seeds the skills table from a CSV file, generating and storing embeddings."""
    print("Seeding skills from CSV...")
    try:
        skills_df = pd.read_csv(skills_csv)
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] Skills CSV file not found at {skills_csv}")
    
    # Ensure 'skill' column exists and clean data
    if "skill" not in skills_df.columns:
        raise ValueError("[ERROR] 'skill' column not found in skills CSV")
    skills = skills_df["skill"].dropna().str.strip().str.lower().unique().tolist()
    
    if not skills:
        print("No valid skills found in CSV. Skipping seeding.")
        return
    
    print(f"Generating embeddings for {len(skills)} skills...")
    try:
        embeddings = embedder.encode(skills, show_progress_bar=True)
    except Exception as e:
        raise SystemExit(f"[ERROR] Failed to generate embeddings: {str(e)}")
    
    with engine.begin() as conn:
        try:
            conn.execute(text("DELETE FROM skills"))
            for skill, emb in zip(skills, embeddings):
                conn.execute(
                    text("INSERT IGNORE INTO skills(name, canonical_name, embedding) VALUES (:name, :cn, :emb)"),
                    {"name": skill, "cn": skill, "emb": json.dumps(emb.tolist())}
                )
            print("✅ Skills table seeded successfully.")
        except Exception as e:
            raise SystemExit(f"[ERROR] Failed to seed skills table: {str(e)}")

def ingest_jobs(raw_csv="data/raw_jobs.csv", sim_threshold=0.25):
    """Ingests jobs from a CSV and links them to skills based on embedding similarity."""
    print("\nStarting job ingestion process...")
    try:
        jobs_df = load_csv_data(raw_csv, ["title", "company_name", "location", "posted_date", "description", "source"])
    except SystemExit as e:
        raise e
    except Exception as e:
        raise SystemExit(f"[ERROR] Failed to load jobs CSV: {str(e)}")
    
    with engine.begin() as conn:
        try:
            # Clear existing data
            conn.execute(text("DELETE FROM job_skills"))
            conn.execute(text("DELETE FROM jobs"))
            conn.execute(text("DELETE FROM companies"))

            # Insert unique companies
            for cname in jobs_df['company_name'].dropna().unique():
                conn.execute(text("INSERT IGNORE INTO companies(name) VALUES (:n)"), {"n": cname})
            company_map = {r['name']: r['company_id'] for r in conn.execute(text("SELECT company_id, name FROM companies")).mappings().all()}

            print("Caching skills and embeddings from database...")
            skills_rows = conn.execute(text("SELECT skill_id, name, embedding FROM skills")).mappings().all()
            processed_skills = [dict(row) for row in skills_rows]
            for skill in processed_skills:
                if skill.get('embedding'):
                    try:
                        skill['emb_array'] = np.array(json.loads(skill['embedding']))
                    except json.JSONDecodeError:
                        print(f"[WARNING] Invalid embedding for skill {skill['name']}. Skipping.")
                        skill['emb_array'] = None

            print(f"Ingesting {len(jobs_df)} jobs...")
            for _, r in jobs_df.iterrows():
                company_id = company_map.get(r['company_name'])
                if not company_id:
                    print(f"[WARNING] Company {r['company_name']} not found for job '{r['title']}'. Skipping.")
                    continue
                
                params = {
                    "title": r['title'], "cid": company_id, "loc": r['location'],
                    "pdate": r['posted_date'], "desc": r['description'], "src": r['source']
                }
                
                job_res = conn.execute(
                    text("INSERT INTO jobs(title, company_id, location, posted_date, description, source) VALUES (:title, :cid, :loc, :pdate, :desc, :src)"),
                    params
                )
                job_id = job_res.lastrowid

                job_text = (str(r.get('title', '')) + " " + str(r.get('description', ''))).lower()
                if not job_text.strip():
                    print(f"  - Skipping skill linking for job ID {job_id} due to empty text.")
                    continue

                try:
                    job_emb = embedder.encode([job_text])[0]
                except Exception as e:
                    print(f"[WARNING] Failed to generate embedding for job ID {job_id} ('{r['title']}'): {str(e)}. Skipping.")
                    continue
                
                skills_linked_count = 0
                for skill in processed_skills:
                    skill_emb = skill.get('emb_array')
                    if skill_emb is None or skill_emb.size == 0:
                        continue

                    try:
                        sim = np.dot(job_emb, skill_emb) / (np.linalg.norm(job_emb) * np.linalg.norm(skill_emb))
                        if sim >= sim_threshold:
                            conn.execute(
                                text("INSERT IGNORE INTO job_skills(job_id, skill_id, weight) VALUES (:j, :s, :w)"),
                                {"j": job_id, "s": skill['skill_id'], "w": float(sim)}
                            )
                            skills_linked_count += 1
                    except Exception as e:
                        print(f"[WARNING] Error computing similarity for skill {skill['name']} and job ID {job_id}: {str(e)}")
                print(f"  - Linked {skills_linked_count} skills to job ID {job_id} ('{r['title']}')")
        except Exception as e:
            raise SystemExit(f"[ERROR] Failed during job ingestion: {str(e)}")

    print("✅ Jobs ingested and linked to skills successfully.")

if __name__ == "__main__":
    seed_skills()  # Run seed_skills first to ensure skills are available
    ingest_jobs()