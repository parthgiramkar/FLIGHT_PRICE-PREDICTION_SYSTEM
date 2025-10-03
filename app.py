from flask import Flask , url_for , render_template
from forms import Inputform

import pandas as pd
import joblib

opy = Flask(__name__)

opy.config["SECRET_KEY"] = "yes,the proj"

model = joblib.load("model.joblib")     # expecting the path of joblib


@opy.route("/")
@opy.route("/home")

def h() :
	return render_template("home.html",title="Home")


# @opy.route("/predict")                # deafult in GET method
@opy.route("/predict" , methods=["GET","POST"])
def pred() :

	fog = Inputform()
	if fog.validate_on_submit() :
		# getting the data from the form
		x_new = pd.DataFrame(dict(
			airline=[fog.airline.data],
			date_of_journey=[fog.date_of_journey.data.strftime("%Y-%m-%d")],
			source =[fog.source.data],
			destination=[fog.destination.data],
			arrival_time=[fog.arrival_time.data.strftime("%H:%M:%S")],
			dep_time=[fog.dep_time.data.strftime("%H:%M:%S")],
			duration=[fog.duration.data],

			total_stops=[fog.total_stops.data],
			additional_info=[fog.additional_info.data],
						
		))

		prediction = model.predict(x_new)[0]       # return an array
		mess = f"The predicted price is {prediction} INR "
	else :
		mess = "Please provide valid input details !"

	return render_template("predict.html",title="Predict " , fo = fog , output=mess)


if __name__ == "__main__" :

	opy.run(debug=True)
