from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Train the model directly from gamer.csv when the app starts.
data = pd.read_csv("gamer.csv")

required_columns = ["hours", "accuracy", "experience", "skill"]
if not all(column in data.columns for column in required_columns):
    raise ValueError(
        "gamer.csv must contain: hours, accuracy, experience, skill"
    )

X = data[["hours", "accuracy", "experience"]]
y = data["skill"]

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=3
)
model.fit(X, y)

print("Gamer skill model trained successfully from gamer.csv!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        hours = float(request.form["hours"])
        accuracy = float(request.form["accuracy"])
        experience = float(request.form["experience"])

        if not 0 <= hours <= 24:
            raise ValueError("Gaming hours must be between 0 and 24.")
        if not 0 <= accuracy <= 100:
            raise ValueError("Accuracy must be between 0 and 100.")
        if experience < 0:
            raise ValueError("Experience cannot be negative.")

        player_data = pd.DataFrame(
            [[hours, accuracy, experience]],
            columns=["hours", "accuracy", "experience"]
        )
        result = model.predict(player_data)[0]

        return render_template("index.html", prediction=result)

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            error="Please enter valid values for all three fields."
        )


if __name__ == "__main__":
    app.run(debug=True)
