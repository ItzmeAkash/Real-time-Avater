# Real-time Avatar

A LiveKit-based real-time avatar application with Flask API server for token generation and a LiveKit agent worker for avatar interactions.

## Architecture

- **server.py**: Flask API server that generates LiveKit access tokens
- **main.py**: LiveKit agent worker that handles avatar sessions using Tavus, Deepgram, and Groq

## Prerequisites

- Docker and Docker Compose installed
- LiveKit server URL and API credentials
- Deepgram API key
- Groq API key
- Tavus API key

## Environment Setup

Create a `.env` file in the root directory with the following variables:

```env
# LiveKit Configuration
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# Service API Keys
DEEPGRAM_API_KEY=your_deepgram_key
GROQ_API_KEY=your_groq_key
TAVUS_API_KEY=your_tavus_key

# Optional: Port configuration
PORT=8080
```

## Docker Setup

### Development Mode

Run both services in development mode using Docker Compose:

```bash
# Start API server and agent in dev mode
docker-compose up api-server agent-dev
```

Or run them in detached mode:

```bash
docker-compose up -d api-server agent-dev
```

### Production Mode

Run in production mode:

```bash
# Start API server and agent in production mode
docker-compose --profile production up api-server agent-prod
```

Or run in detached mode:

```bash
docker-compose --profile production up -d api-server agent-prod
```

### Individual Services

You can also run services individually:

**API Server only:**
```bash
docker-compose up api-server
```

**Agent in dev mode only:**
```bash
docker-compose up agent-dev
```

**Agent in production mode only:**
```bash
docker-compose --profile production up agent-prod
```

## Building Docker Images

Build all images:

```bash
docker-compose build
```

Build specific service:

```bash
# API Server
docker-compose build api-server

# Agent Dev
docker-compose build agent-dev

# Agent Prod
docker-compose --profile production build agent-prod
```

## Running Without Docker Compose

### API Server

```bash
# Build
docker build -f Dockerfile.api -t realtime-avatar-api .

# Run
docker run -p 8080:8080 --env-file .env realtime-avatar-api
```

### Agent in Dev Mode

```bash
# Build
docker build -f Dockerfile.dev -t realtime-avatar-agent-dev .

# Run
docker run --env-file .env realtime-avatar-agent-dev
```

### Agent in Production Mode

```bash
# Build
docker build -f Dockerfile -t realtime-avatar-agent .

# Run
docker run --env-file .env realtime-avatar-agent
```

## API Endpoints

Once the API server is running:

- **GET/POST `/getToken`**: Generate a LiveKit access token
  - Query params/JSON body:
    - `identity` (optional): User identity, default "user"
    - `name` (optional): User name, default "User"
    - `room` (optional): Room name, default "my-room"

- **GET `/health`**: Health check endpoint

## Example API Usage

```bash
# Get token via GET request
curl "http://localhost:8080/getToken?identity=user1&name=John&room=test-room"

# Get token via POST request
curl -X POST http://localhost:8080/getToken \
  -H "Content-Type: application/json" \
  -d '{"identity": "user1", "name": "John", "room": "test-room"}'
```

## Development

### Local Development (Without Docker)

**Run API Server:**
```bash
python server.py
```

**Run Agent in Dev Mode:**
```bash
python main.py dev
```

**Run Agent in Production Mode:**
```bash
python main.py start
```

## Logs

View logs from all services:

```bash
docker-compose logs -f
```

View logs from specific service:

```bash
# API Server logs
docker-compose logs -f api-server

# Agent dev logs
docker-compose logs -f agent-dev

# Agent prod logs
docker-compose logs -f agent-prod
```

## Stopping Services

Stop all running services:

```bash
docker-compose down
```

Stop and remove volumes:

```bash
docker-compose down -v
```

## Troubleshooting

1. **Agent not connecting**: Check that LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are correctly set in `.env`

2. **API server not starting**: Verify PORT is not already in use and check environment variables

3. **Avatar not working**: Ensure TAVUS_API_KEY, DEEPGRAM_API_KEY, and GROQ_API_KEY are set correctly

4. **Build failures**: Make sure all dependencies in `requirements.txt` are compatible and the Dockerfile is using the correct Python version

## File Structure

```
.
├── Dockerfile              # Production agent worker
├── Dockerfile.api         # API server
├── Dockerfile.dev         # Development agent worker
├── docker-compose.yml     # Docker Compose configuration
├── main.py                # LiveKit agent worker
├── server.py              # Flask API server
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (create this)
```
