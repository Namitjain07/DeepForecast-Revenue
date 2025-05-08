from flask import Flask, request, jsonify
import time

app = Flask(__name__)

@app.route('/v1/completions', methods=['POST'])
def completions():
    data = request.json
    prompt = data.get("prompt", "")
    
    # Simulate server processing delay
    time.sleep(10)  # Adjust the delay as needed
    
    response = {
        "output": f"Received prompt: '{prompt}' - This is the server's response."
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=1234, debug=True)