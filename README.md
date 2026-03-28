#  CardioPredict AI

CardioPredict AI is a Streamlit-based web application that uses Machine Learning to predict the risk of heart disease based on key health parameters. The application provides real-time predictions along with meaningful insights to support early detection and preventive care.


## 🚀 Features

* 🔍 Predicts heart disease risk using ML model
* 📊 Displays probability with risk level (Low / Medium / High)
* 📈 Visual insights using interactive charts
* 🧠 Highlights top contributing health factors
* 💡 Provides personalized health suggestions
* 🎨 Clean and modern UI with responsive design

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Joblib


## 📂 Project Structure

```
heart-disease/
│── app.py
│── requirements.txt
│── heart_best_model.pkl
│── *.png (images)
```


## ⚙️ Installation & Running Locally

1. Clone the repository:

```
git clone https://github.com/SMounicaVaishnaviShakuntala/heart-disease.git
```

2. Navigate to the project folder:

```
cd heart-disease
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the app:

```
streamlit run app.py
```


## 🌐 Deployment

This app is deployed using Streamlit Community Cloud.

👉 Live App: https://heart-disease-prediction2026.streamlit.app/


## 📊 Input Parameters

The model takes the following inputs:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Maximum Heart Rate Achieved


## 🧠 How It Works

* The trained Machine Learning model analyzes user inputs
* It calculates the probability of heart disease
* Based on probability, risk is categorized into:

  * Low Risk
  * Medium Risk
  * High Risk
* Feature importance is used to generate insights


## ⚠️ Disclaimer

This application is for educational and informational purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment.


## 👩‍💻 Author

Vaishnavi


## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
