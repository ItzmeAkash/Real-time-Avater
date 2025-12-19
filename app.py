from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_routes import router

app = FastAPI(
    title="LiveKit Avatar API",
    description="API for LiveKit token generation and conversation evaluation",
    version="1.0.0",
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router
app.include_router(router)


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "LiveKit Avatar API",
        "endpoints": {
            "getToken": "/getToken (GET/POST) - Get LiveKit token",
            "evaluate": "/evaluate (POST) - Evaluate a transcript",
            "transcript": "/transcript/{room_name} (GET) - Get transcript for a room",
            "evaluation": "/evaluation/{room_name} (GET) - Get evaluation for a room",
            "evaluate_room": "/evaluate/{room_name} (POST) - Get transcript and evaluate a room",
            "health": "/health (GET) - Health check",
            "docs": "/docs - API documentation",
        },
    }
