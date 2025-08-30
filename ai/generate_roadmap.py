import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")

# --- Pydantic Data Structures for Structured Output ---
# These models define the exact JSON structure we want the AI to return.

class Goal(BaseModel):
    goal: str = Field(description="A specific, actionable goal for the period.")
    resources: list[str] = Field(description="A list of 1-2 concrete resources (e.g., course name, book title, URL).")

class PlanPeriod(BaseModel):
    period: str = Field(description="The specific period for this plan segment, e.g., 'Week 1' or 'Days 1-15'.")
    focus: str = Field(description="The primary focus for this period.")
    tasks: list[Goal] = Field(description="A list of specific goals for the period.")

class Roadmap(BaseModel):
    skill: str = Field(description="The skill this roadmap is for.")
    plan: list[PlanPeriod] = Field(description="A list of plans for each period, covering the entire requested duration.")

# --- LangChain Setup ---
llm = ChatGroq(model_name=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.2)

# The prompt now includes a dynamic 'duration' parameter
prompt_template = """
You are a practical and motivating career coach. Your task is to create a detailed, actionable learning roadmap for a user who wants to learn a new skill. The total duration for this roadmap is {duration}.

**User's Goal:** {user_summary}
**Skill to Learn:** {skill_name} (Importance Score: {importance})

Create a comprehensive plan for the specified duration, broken down into logical periods (e.g., weekly for shorter plans, bi-weekly for longer ones). For each period, provide a clear focus and 2-3 specific, actionable goals. For each goal, suggest 1-2 concrete learning resources (like a specific online course, a book, a tutorial URL, or a type of mini-project).

**Constraints:**
- The plan must be structured and realistic for the given duration.
- Prioritize foundational knowledge first.
- The output MUST be in the requested JSON format.
"""

# --- Core Roadmap Generation Functions ---
def create_single_roadmap(skill_name, importance, user_summary, duration):
    """Generates a roadmap for a single skill using the LLM and structured output."""
    try:
        parser = JsonOutputParser(pydantic_object=Roadmap)
        
        prompt = ChatPromptTemplate.from_template(
            template=prompt_template,
            template_format="f-string",
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | llm | parser
        
        # Pass the dynamic duration to the chain
        result = chain.invoke({
            "skill_name": skill_name,
            "importance": importance,
            "user_summary": user_summary,
            "duration": duration
        })
        return result
    except Exception as e:
        print(f"Error generating roadmap for '{skill_name}': {e}")
        return {"error": f"Failed to generate roadmap for {skill_name}", "details": str(e)}

def generate_roadmap(user_summary, missing_skills, duration):
    """
    Generates learning roadmaps for the top 3 missing skills in parallel for a specified duration.
    """
    roadmaps = []
    skills_to_plan = missing_skills[:3]
    
    with ThreadPoolExecutor(max_workers=len(skills_to_plan)) as executor:
        # Pass the duration to each parallel task
        futures = [
            executor.submit(create_single_roadmap, name, imp, user_summary, duration)
            for name, imp in skills_to_plan
        ]
        for future in as_completed(futures):
            roadmaps.append(future.result())
            
    return roadmaps

