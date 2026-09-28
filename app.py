from flask import Flask, render_template, request
from nlp_pipeline import process_text


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    text = ""

    if request.method == "POST":

        text = request.form.get("text", "").strip()

        if text:
            result = process_text(text)

    return render_template(
        "index.html",
        result=result,
        text=text
    )


if __name__ == "__main__":
    app.run(debug=True)