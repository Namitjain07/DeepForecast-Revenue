from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        user_input = request.form["prompt"]
        url = "http://127.0.0.1:1234/v1/completions"
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "prompt": user_input,
            "max_tokens": 100,
            "temperature": 0.7
        }
        try:
            # Increase timeout to allow server enough time to respond
            response = requests.post(url, headers=headers, json=data, verify=False, timeout=30)
            if response.status_code == 200:
                output = response.json().get("output", "No response received.")
            else:
                output = f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            output = f"Exception: {str(e)}"
        
        return render_template("chatbot.html", output=output)
    return render_template("chatbot.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)