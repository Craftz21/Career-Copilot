from fastapi import APIRouter

router = APIRouter(prefix="/v1/roles", tags=["roles"])

_FALLBACK_ROLES = [
    "Software Engineer",
    "Backend Software Engineer",
    "Frontend Developer",
    "Full Stack Developer",
    "Data Scientist",
    "Machine Learning Engineer",
    "Product Manager",
    "DevOps Engineer",
    "QA Engineer",
]


@router.get("", response_model=list[str])
async def get_roles():
    return _FALLBACK_ROLES
