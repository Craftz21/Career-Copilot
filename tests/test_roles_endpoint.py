from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.roles import router as roles_router


def test_roles_endpoint_returns_available_target_roles():
    app = FastAPI()
    app.include_router(roles_router)

    with TestClient(app) as client:
        response = client.get("/v1/roles")

    assert response.status_code == 200
    assert response.json() == [
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
