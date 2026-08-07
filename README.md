# 🏠 NYC Airbnb Room Type Predictor

A Machine Learning web application that predicts the **room type** of a New York City Airbnb listing based on property details such as location, price, availability, reviews, and host information.

## 🚀 Live Demo

🌐 https://nyc-airbnb-room-type-predictor-3-h07y.onrender.com

---

## 📖 Overview

This project uses a trained **Scikit-learn Machine Learning model** served through **FastAPI** to classify Airbnb listings into different room types.

Users simply enter listing details, and the application instantly predicts the most likely room type with a clean, responsive web interface.

---

## ✨ Features

- 🏠 Predict Airbnb Room Type
- ⚡ FastAPI REST API
- 🎨 Responsive HTML, CSS & JavaScript UI
- 🤖 Machine Learning using Scikit-learn
- 📊 Data preprocessing pipeline
- ☁️ Deployed on Render

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- FastAPI
- Uvicorn

### Machine Learning
- Scikit-learn
- Pandas
- Joblib

---

## 📂 Project Structure

```text
NYC-Airbnb-Room-Type-Predictor/
│
├── main.py
├── index.html
├── style.css
├── script.js
├── requirements.txt
├── runtime.txt
├── Model_Pipeline.pkl
├── AB_NYC_2019.csv
└── README.md
```

---

## 📊 Dataset

Dataset Used:

**AB_NYC_2019.csv**

Input Features:

- Latitude
- Longitude
- Borough
- Neighbourhood
- Price
- Minimum Nights
- Number of Reviews
- Reviews per Month
- Availability (365)
- Host Listings Count

Target Variable:

- Room Type

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/arjunverma000005-hue/NYC-Airbnb-Room-Type-Predictor.git
```

Move into the project directory

```bash
cd NYC-Airbnb-Room-Type-Predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
uvicorn main:app --reload
```

Open your browser

```
http://127.0.0.1:8000
```

---

## 📦 Requirements

- FastAPI
- Uvicorn
- Pandas
- Scikit-learn
- Joblib

---

## 🌍 Live Deployment

**Live Application**

https://nyc-airbnb-room-type-predictor-3-h07y.onrender.com

---

## 👨‍💻 Developer

**Arjun Verma**

GitHub:
https://github.com/arjunverma000005-hue

---

## ⭐ If you like this project

Please give this repository a ⭐ on GitHub.

---

## 📜 License

This project is for educational and portfolio purposes.
