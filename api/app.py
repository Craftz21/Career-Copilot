import os
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
# This is the line that was missing
from sqlalchemy import create_engine, text
from analysis.resume_analyzer import parse_pdf_text, extract_skills_from_text, save_user_skills, compute_gap
from ai.generate_roadmap import generate_roadmap

# Load environment variables from .env file at the start
load_dotenv()

# --- Database configuration ---
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASS = os.getenv("MYSQL_PASS", "your_mysql_pass")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_DB   = os.getenv("MYSQL_DB", "career_copilot")

engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}/{MYSQL_DB}")

# --- FastAPI setup ---
app = FastAPI(title="Career Copilot")

# --- Endpoints ---
@app.post("/upload-resume", response_model=dict)
async def upload_resume(name: str = Form(...), email: str = Form(...), file: UploadFile = File(...)):
    # 1) create/find user
    with engine.begin() as conn:
        conn.execute(
            text("INSERT IGNORE INTO users(name, email) VALUES (:n, :e)"),
            {"n": name, "e": email}
        )
        row = conn.execute(
            text("SELECT user_id FROM users WHERE email=:e"),
            {"e": email}
        ).mappings().first()
        user_id = row["user_id"]

    # 2) save + parse PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    parsed_text = parse_pdf_text(tmp_path)
    os.unlink(tmp_path)  # cleanup temp file

    # 3) persist resume
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO resumes(user_id, filename, parsed_text) VALUES (:u, :fn, :txt)"),
            {"u": user_id, "fn": file.filename, "txt": parsed_text}
        )

    # 4) extract skills and save
    matched = extract_skills_from_text(parsed_text, top_k=30)
    save_user_skills(user_id, matched)

    return {"user_id": user_id, "matched_skills": [{"name": m[1], "score": m[2]} for m in matched]}

@app.get("/gap/{user_id}")
def gap(user_id: int, role: str):
    missing = compute_gap(user_id, role)
    return {"missing_skills": [{"skill": m[1], "importance": m[2]} for m in missing]}

@app.post("/roadmap/{user_id}")
def roadmap(user_id: int, role: str, summary: str = "", duration: str = "90 days"):
    """
    Generates a learning roadmap for a user.
    - user_id: The ID of the user.
    - role: The target job role.
    - summary: A brief summary of the user's goals.
    - duration: The desired length of the roadmap (e.g., "2 weeks", "1 month"). Defaults to "90 days".
    """
    missing = compute_gap(user_id, role)
    if not missing:
        return {"roadmap": [], "message": "No missing skills found for this role."}

    # Format skills: (name, importance)
    missing_formatted = [(m[1], m[2]) for m in missing[:3]]  # Top 3 skills for detailed plans
    user_summary = summary or f"A user targeting the role of {role}"

    # --- Generate roadmap via Groq ---
    plan = generate_roadmap(user_summary, missing_formatted, duration)

    # 5) save recommendation
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO recommendations(user_id, summary, details) VALUES (:u, :s, :d)"),
            {"u": user_id, "s": f"Roadmap for {role} ({duration})", "d": json.dumps(plan)}
        )

    return {"roadmap": plan}

