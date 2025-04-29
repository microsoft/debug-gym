import json

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load the JSON file (replace 'your_file.json' with the actual filename)
with open(
    "/home/macote/src/pdb/exps/swebench-lite/pdb_agent_4o-mini_0/django__django-10924/froggy.jsonl",
    "r",
) as f:
    data = json.load(f)


def to_pretty_json(value):
    return json.dumps(value, sort_keys=True, indent=4, separators=(",", ": "))


app.jinja_env.filters["tojson_pretty"] = to_pretty_json


@app.route("/")
def index():
    # Pass metadata to the template
    metadata = {
        "problem": data["problem"],
        "config": data["config"],
        "uuid": data["uuid"],
        "success": data["success"],
    }
    total_steps = len(data["log"])
    return render_template("index.html", metadata=metadata, total_steps=total_steps)


@app.route("/get_step/<int:step_id>")
def get_step(step_id):
    # Return the specific step data as JSON
    if 0 <= step_id < len(data["log"]):
        step = data["log"][step_id]
        return jsonify(step)
    return jsonify({"error": "Step not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
