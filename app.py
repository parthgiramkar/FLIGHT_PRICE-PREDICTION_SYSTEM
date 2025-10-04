from flask import Flask, url_for, render_template
from forms import Inputform
import pandas as pd
import joblib

# RENAMED 'opy' to 'app'
app = Flask(__name__)

app.config["SECRET_KEY"] = "yes,the proj"

model = joblib.load("model.joblib")  # expecting the path of joblib


# RENAMED '@opy.route' to '@app.route'
@app.route("/")
@app.route("/home")
def h():
    return render_template("home.html", title="Home")


# RENAMED '@opy.route' to '@app.route'
@app.route("/predict", methods=["GET", "POST"])
def pred():
    fog = Inputform()
    if fog.validate_on_submit():
        x_new = pd.DataFrame(dict(
            airline=[fog.airline.data],
            date_of_journey=[fog.date_of_journey.data.strftime("%Y-%m-%d")],
            source=[fog.source.data],
            destination=[fog.destination.data],
            arrival_time=[fog.arrival_time.data.strftime("%H:%M:%S")],
            dep_time=[fog.dep_time.data.strftime("%H:%M:%S")],
            duration=[fog.duration.data],
            total_stops=[fog.total_stops.data],
            additional_info=[fog.additional_info.data],
        ))

        prediction = model.predict(x_new)[0]
        mess = f"The predicted price is {prediction} INR"
    else:
        mess = "Please provide valid input details!"

    return render_template("predict.html", title="Predict", fo=fog, output=mess)


if __name__ == "__main__":
    # RENAMED 'opy.run' to 'app.run'
    app.run(debug=True)
