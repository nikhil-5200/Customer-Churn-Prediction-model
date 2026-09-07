# 📊 Customer Churn Prediction

A **Deep Learning project** that predicts whether a telecom customer is likely to churn based on customer demographics, services, contract details, and billing information.

## 🎯 Objective

The goal of this project is to identify customers who are likely to leave a telecom service so that businesses can take preventive retention actions.

## 📂 Dataset

**Telco Customer Churn Dataset**

* 7,043 customer records
* 21 original features
* Target variable: `Churn`
* `1` = Churn
* `0` = No Churn

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow / Keras

## 🔄 Project Workflow

```text
Data Collection
      ↓
Data Cleaning
      ↓
Feature Encoding
      ↓
Feature Scaling
      ↓
Train-Test Split
      ↓
Neural Network
      ↓
Model Evaluation
      ↓
Churn Prediction
```

## 🧹 Data Preprocessing

The project includes:

* Removing `customerID`
* Handling missing `TotalCharges` values
* Converting Yes/No values into numerical values
* One-hot encoding categorical features
* Scaling numerical features using `MinMaxScaler`

## 🤖 Model

A simple **Artificial Neural Network (ANN)** is built using TensorFlow/Keras.

```text
Input Layer
     ↓
Dense Layer (10 neurons, ReLU)
     ↓
Output Layer (1 neuron, Sigmoid)
```

**Optimizer:** Adam
**Loss:** Binary Crossentropy
**Training:** 100 epochs

## 📈 Evaluation

The dataset is divided into:

* **80% Training**
* **20% Testing**

The model is evaluated on unseen test data to measure its prediction performance.

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── customer churn prediction.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

## 🚀 How to Run

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

Run the Python file:

```bash
python "customer churn prediction.py"
```

## 💡 Key Outcome

This project demonstrates how **machine learning and deep learning can be used to predict customer churn and support customer retention strategies.**

## 👨‍💻 Author

Nikhil

Aspiring **Data Analyst** passionate about transforming raw data into meaningful business insights.

- 💼 LinkedIn:(https://www.linkedin.com/in/nikhil-kaushik-476337422?utm_source=share_via&utm_content=profile&utm_medium=member_ios)
- 📧 Email: kaushnikhil@gmail.com
- 🌐 GitHub: https://github.com/nikhil-5200

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
