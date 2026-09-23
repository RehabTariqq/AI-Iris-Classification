# AI Iris Classification 🌸

A basic supervised machine learning project that classifies Iris flowers using the **K-Nearest Neighbors (KNN)** algorithm.

**Built as Project 2 of my Artificial Intelligence Internship at DecodeLabs.**

##  Objective

* Load and understand a dataset
* Split data into training and testing sets
* Train a classification model
* Make predictions
* Evaluate model performance

##  Dataset

The project uses the **Iris dataset** from `scikit-learn`.

* 150 samples
* 4 features
* 3 classes:

  * Setosa
  * Versicolor
  * Virginica

### Features

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

##  Technologies

* Python
* Scikit-learn
* Pandas
* Matplotlib
* K-Nearest Neighbors (KNN)

##  Workflow

```text
Iris Dataset
     ↓
Feature Scaling
     ↓
80/20 Train-Test Split
     ↓
KNN Classification
     ↓
Predictions
     ↓
Accuracy + F1 Score
     ↓
Confusion Matrix
```

##  Evaluation

The model is evaluated using:

* **Accuracy**
* **F1 Score**
* **Confusion Matrix**

Results will be added after running the final model.

## 📂 Project Structure

```text
AI-Iris-Classification/
│
├── iris_classifier.py
├── confusion_matrix.png
├── README.md
└── .gitignore
```

##  How to Run

### 1. Clone the repository

```bash
git clone https://github.com/RehabTariqq/AI-Iris-Classification.git
cd AI-Iris-Classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install scikit-learn pandas matplotlib
```

### 5. Run the project

```powershell
python iris_classifier.py
```

##  What I Learned

* Basics of supervised learning
* Dataset handling
* Feature scaling
* Train-test splitting
* KNN classification
* Model evaluation
* Confusion matrices and F1 scores

##  Author

**Rehab Tariq**
Computer Science Student

*Built as part of my Artificial Intelligence Internship at DecodeLabs.*
