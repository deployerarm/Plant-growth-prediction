from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import torch
import numpy as np
import joblib
from werkzeug.utils import secure_filename
from gradcam import load_model, GradCAM, predict_image, generate_gradcam
import matplotlib.pyplot as plt
from Models.llm_model import explain_growth_llm
import sqlite3
import re

# ----------------------------------------------------
# Flask Setup
# ----------------------------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------------------------------------------
# Load Image Classification Model
# ----------------------------------------------------
model_path = "Models/mobilenet.pth"
clf_model = load_model(model_path)
target_layer = clf_model.conv_head
gradcam = GradCAM(clf_model, target_layer)

# ----------------------------------------------------
# Load Regression + Encoder + Scaler
# ----------------------------------------------------
reg_model = joblib.load("Models/growth_pred_model.sav")
le = joblib.load("Models/label_encoder.sav")
scaler = joblib.load("Models/lstm_scaler.sav")

# ----------------------------------------------------
# Define LSTM Forecast Model
# ----------------------------------------------------
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load("Models/growth_forecast_lstm.pth", map_location="cpu"))
lstm_model.eval()

# ----------------------------------------------------
# Multi-step Forecast Function
# ----------------------------------------------------
def forecast_future_height(
        model, scaler, temperature, humidity, soil_moisture,
        soil_ph, light_intensity, current_height, days=60):

    base_raw = np.array([
        temperature, humidity, soil_moisture,
        soil_ph, light_intensity, current_height
    ])

    seq_raw = np.tile(base_raw, (30, 1))
    seq_scaled = scaler.transform(seq_raw)

    preds_raw = []

    h_min = scaler.data_min_[5]
    h_max = scaler.data_max_[5]
    h_range = max(h_max - h_min, 1e-8)

    for _ in range(days):
        inp = torch.tensor(seq_scaled[np.newaxis, :, :], dtype=torch.float32)

        with torch.no_grad():
            next_scaled = float(model(inp).item())

        next_raw_height = next_scaled * h_range + h_min
        preds_raw.append(next_raw_height)

        new_raw = np.array([
            temperature, humidity, soil_moisture,
            soil_ph, light_intensity, next_raw_height
        ])
        new_scaled = scaler.transform(new_raw.reshape(1, -1))[0]

        seq_scaled = np.vstack([seq_scaled[1:], new_scaled])

    return preds_raw

# ----------------------------------------------------
# PAGE 1 — Upload Page
# ----------------------------------------------------
@app.route("/classify", methods=["GET", "POST"])
def classify():
    if request.method == "POST":
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        label, confidence = predict_image(clf_model, filepath)

        _, _, cam_img = generate_gradcam(clf_model, gradcam, filepath)
        cam_path = os.path.join(app.config["UPLOAD_FOLDER"], "cam_" + filename)
        cv2.imwrite(cam_path, cam_img)

        return redirect(url_for("input_form",
                                plant=label,
                                filename=filename,
                                cam="cam_" + filename))
    return render_template("home.html")

# ----------------------------------------------------
# PAGE 2 — Input Form
# ----------------------------------------------------
@app.route("/input_form")
def input_form():
    return render_template(
        "input_form.html",
        plant=request.args.get("plant"),
        filename=request.args.get("filename"),
        cam=request.args.get("cam"),
    )

# ----------------------------------------------------
# PAGE 3 — Final Prediction + Forecast
# ----------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    plant_species = request.form["plant_species"]
    plant_age = float(request.form["plant_age_days"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    soil_moisture = float(request.form["soil_moisture"])
    soil_ph = float(request.form["soil_ph"])
    light_intensity = float(request.form["light_intensity"])

    plant_species_label = le.transform([plant_species])[0]

    X = np.array([[
        plant_species_label, plant_age, temperature,
        humidity, soil_moisture, soil_ph, light_intensity
    ]])

    height, leaf_area, health_score = reg_model.predict(X)[0]

    # Forecasts
    forecast_60 = forecast_future_height(lstm_model, scaler,
                                         temperature, humidity, soil_moisture,
                                         soil_ph, light_intensity, height, days=60)

    forecast_120 = forecast_future_height(lstm_model, scaler,
                                          temperature, humidity, soil_moisture,
                                          soil_ph, light_intensity, height, days=120)


    # LLM Explanation
    llm_text = explain_growth_llm(
        plant_species,
        height,
        leaf_area,
        health_score,
        forecast_60,
        forecast_120,
    )

    # Plot forecast graph
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, 61), forecast_60, linewidth=2)
    plt.xlabel("Days Ahead")
    plt.ylabel("Predicted Height (cm)")
    plt.title(f"60-Day Growth Forecast — {plant_species}")
    plt.grid(True)

    graph_path = os.path.join(app.config["UPLOAD_FOLDER"], "forecast_plot.png")
    plt.savefig(graph_path)
    plt.close()

    return render_template(
        "results.html",
        plant=plant_species,
        height=round(float(height), 2),
        leaf_area=round(float(leaf_area), 2),
        health=round(float(health_score), 2),
        forecast_next=round(float(forecast_60[0]), 2),
        forecast_60=forecast_60,
        forecast_120=forecast_120,
        graph="forecast_plot.png",
        llm_explanation=llm_text
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()

        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    

        elif mail1 == 'admin' and password1 == 'admin':
            return render_template("home.html")

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/graphs1')
def home1():
	return render_template('graphs1.html')

@app.route('/graphs2')
def graphs2():
	return render_template('graphs2.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')


if __name__ == "__main__":
    app.run(debug=True)
