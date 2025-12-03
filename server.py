import os
from livekit import api
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


@app.route("/getToken", methods=["GET", "POST"])
def getToken():
    try:
        # Get parameters from query string or JSON body
        if request.method == "POST":
            data = request.get_json() or {}
            identity = data.get("identity", "user")
            name = data.get("name", "User")
            room = data.get("room", "my-room")
        else:
            identity = request.args.get("identity", "user")
            name = request.args.get("name", "User")
            room = request.args.get("room", "my-room")

        # Get API credentials from environment
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")

        if not api_key or not api_secret:
            return jsonify({"error": "LiveKit API credentials not configured"}), 500

        # Generate token
        token = (
            api.AccessToken(api_key, api_secret)
            .with_identity(identity)
            .with_name(name)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
        )

        return jsonify({
            "token": token.to_jwt(),
            "url": os.getenv("LIVEKIT_URL", ""),
            "room": room,
            "identity": identity,
            "name": name,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "livekit_configured": bool(
            os.getenv("LIVEKIT_API_KEY") and os.getenv("LIVEKIT_API_SECRET")
        ),
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
