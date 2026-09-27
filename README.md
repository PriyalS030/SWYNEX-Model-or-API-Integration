# 🎓 Student Performance Risk Prediction

**SWYNEX Technologies — AI Internship Task 2: Model & API Integration**

---

## 📋 Project Overview

This project is a web-based machine-learning prototype that predicts whether a student is **At Risk** or **Low Risk** of academic underperformance, based on academic, behavioral, and demographic factors.

Built as Task 2 of the SWYNEX Technologies AI Internship, this project demonstrates:
- End-to-end ML pipeline: data loading, preprocessing, training, evaluation, and deployment
- Model comparison and evidence-based selection
- Flask-based web interface for real-time predictions
- Clean architecture for future external AI API integration

> **⚠️ Educational Prototype:** This is an educational project and should not be used for real academic decisions. See [Ethical Considerations](#-ethical-considerations).

---

## 🎯 Problem Statement

Predict whether a student is at risk of academic underperformance based on study habits, past performance, support systems, and behavioral factors. Early identification of at-risk students can help educators provide timely interventions.

---

## 📊 Dataset

**UCI Student Performance Dataset** (Mathematics subject)

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Citation:** P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance." In A. Brito and J. Teixeira (Eds.), Proceedings of 5th Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.
- **Samples:** 395 students
- **Attributes:** 33 (academic, demographic, social, school-related)
- **Subject:** Mathematics (`student-mat.csv`)

---

## 🔧 Features Used

| Feature | Description | Type | Values |
|---------|-------------|------|--------|
| `studytime` | Weekly study time | Ordinal | 1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h |
| `failures` | Number of past class failures | Numeric | 0-3 (3 = 3 or more) |
| `absences` | Number of school absences | Numeric | 0-93 |
| `schoolsup` | Extra educational support | Binary | yes/no |
| `famsup` | Family educational support | Binary | yes/no |
| `paid` | Extra paid classes | Binary | yes/no |
| `activities` | Extracurricular activities | Binary | yes/no |
| `higher` | Wants to pursue higher education | Binary | yes/no |
| `internet` | Internet access at home | Binary | yes/no |
| `freetime` | Free time after school | Ordinal | 1 (very low) to 5 (very high) |
| `goout` | Going out with friends | Ordinal | 1 (very low) to 5 (very high) |
| `health` | Current health status | Ordinal | 1 (very bad) to 5 (very good) |
| `G1` | First period grade | Numeric | 0-20 |
| `G2` | Second period grade | Numeric | 0-20 |

---

## 🎯 Target Variable

| Condition | Label | Encoded |
|-----------|-------|---------|
| G3 < 10 | At Risk | 1 |
| G3 ≥ 10 | Low Risk | 0 |

**Distribution:** 130 At Risk (32.9%) / 265 Low Risk (67.1%)

The threshold of G3 = 10 is a project-specific design choice.

---

## 🤖 AI Approach

- **Task:** Binary classification
- **Preprocessing:** StandardScaler normalization, binary encoding of yes/no features
- **Split:** 80% training (316 samples), 20% testing (79 samples) with stratification
- **Evaluation:** Accuracy, Precision, Recall, F1-score (focused on At Risk class), 5-fold cross-validation

---

## 📈 Models Tested

1. **Logistic Regression** (max_iter=1000)
2. **Decision Tree Classifier**
3. **Random Forest Classifier** (100 estimators)
4. **Support Vector Machine** (RBF kernel)

---

## 📊 Actual Evaluation Results

### Model A: With G1 & G2 Features

| Model | Accuracy | Precision (At Risk) | Recall (At Risk) | F1 (At Risk) | CV F1 Mean |
|-------|----------|---------------------|-------------------|--------------|------------|
| Logistic Regression | 0.9114 | 0.8519 | **0.8846** | 0.8679 | 0.8174 |
| Decision Tree | 0.8608 | 0.7778 | 0.8077 | 0.7925 | 0.7768 |
| **Random Forest** | **0.9241** | **0.9167** | 0.8462 | **0.8800** | **0.8636** |
| SVM | 0.8608 | 0.8000 | 0.7692 | 0.7843 | 0.7942 |

### Model B: Without G1 & G2 Features

| Model | Accuracy | Precision (At Risk) | Recall (At Risk) | F1 (At Risk) | CV F1 Mean |
|-------|----------|---------------------|-------------------|--------------|------------|
| Logistic Regression | 0.7089 | 0.6154 | 0.3077 | 0.4103 | 0.3713 |
| **Decision Tree** | **0.7595** | **0.6522** | **0.5769** | **0.6122** | 0.4261 |
| Random Forest | 0.7089 | 0.5882 | 0.3846 | 0.4651 | 0.4173 |
| SVM | 0.7089 | 0.6667 | 0.2308 | 0.3429 | 0.4214 |

### Key Observations

- **Model A (with G1 & G2) significantly outperforms Model B** across all metrics
- **Random Forest** achieves the highest F1-score (0.88) for the At Risk class in Model A
- **Logistic Regression** has the highest Recall (0.8846) in Model A, meaning it catches the most at-risk students
- Without G1/G2, all models struggle significantly, particularly with Recall for the At Risk class
- The performance gap (F1: 0.88 vs 0.61) demonstrates how strongly G1 and G2 predict G3

---

## ✅ Selected Model

**Random Forest (Model A — With G1 & G2)**

**Reason:** Highest F1-score (0.88) for the At Risk class, combining strong Precision (0.92) and good Recall (0.85). Also has the highest overall Accuracy (0.92) and the best cross-validation F1 mean (0.86), indicating stable generalization.

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher
- pip

### Setup

```bash
# 1. Clone or download the project
cd SWYNEX-Model-or-API-Integration

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates model.pkl and evaluation results)
python train_model.py

# 4. Start the Flask web app
python app.py

# 5. Open in browser
# http://127.0.0.1:5000
```

---

## 💡 Example Input/Output

### Example 1: At Risk Student

| Feature | Value |
|---------|-------|
| Study Time | 1 (< 2 hours) |
| Failures | 2 |
| Absences | 15 |
| School Support | No |
| Family Support | No |
| Paid Classes | No |
| Activities | No |
| Higher Education | No |
| Internet | Yes |
| Free Time | 4 (High) |
| Going Out | 5 (Very High) |
| Health | 3 (Average) |
| G1 | 5 |
| G2 | 5 |

**→ Prediction: ⚠ At Risk**

### Example 2: Low Risk Student

| Feature | Value |
|---------|-------|
| Study Time | 3 (5-10 hours) |
| Failures | 0 |
| Absences | 2 |
| School Support | No |
| Family Support | Yes |
| Paid Classes | Yes |
| Activities | Yes |
| Higher Education | Yes |
| Internet | Yes |
| Free Time | 3 (Medium) |
| Going Out | 2 (Low) |
| Health | 4 (Good) |
| G1 | 14 |
| G2 | 14 |

**→ Prediction: ✓ Low Risk**

---

## ⚖️ Ethical Considerations

1. **Not a definitive judgment:** This model provides a statistical prediction based on limited data. It should never be used as the sole basis for decisions about a student's future, ability, or potential.

2. **Bias and fairness:** The training data comes from two Portuguese schools and may contain biases related to demographics, socioeconomic status, or cultural context that do not generalize to other populations.

3. **Privacy:** Student academic and behavioral data is sensitive. Any real-world deployment would require strict data protection measures and informed consent.

4. **Labeling concern:** The "At Risk" label is based on an arbitrary threshold (G3 < 10) and should not be interpreted as a judgment of the student's character or intelligence.

5. **Self-fulfilling prophecy:** If used in practice, risk labels could negatively affect how educators perceive and interact with students, potentially reinforcing the predicted outcome.

6. **Limited features:** Many important factors (home environment, mental health, learning disabilities, teacher quality, etc.) are not captured in this dataset.

---

## ⚠️ Limitations

1. **Small dataset:** Only 395 students from two Portuguese schools. Results may not generalize to other populations, countries, or educational systems.

2. **G1 and G2 dominance:** First and second period grades (G1, G2) are strongly correlated with the final grade (G3). Including them gives high accuracy but limits the model's practical utility for early-stage intervention (before G1/G2 are available). Without G1/G2, model performance drops significantly (F1: 0.88 → 0.61).

3. **Cross-sectional data:** The dataset captures a snapshot in time. Student performance is dynamic and can change over the academic year.

4. **Binary classification:** The At Risk / Low Risk classification oversimplifies a continuous spectrum of academic performance.

5. **No temporal validation:** The model is evaluated using a random train/test split, not validated across different academic years or cohorts.

---

## 🔮 Future Scope

1. **External AI API integration:** Add natural-language explanations of predictions using an external AI service (architecture is ready — see below)
2. **Feature importance visualization:** Display which factors most influenced the prediction
3. **Model without G1/G2:** Offer an early-warning mode that doesn't require period grades
4. **Larger datasets:** Train on more diverse and larger student populations
5. **Multi-class prediction:** Predict risk levels (Low / Medium / High) instead of binary classification
6. **Explainable AI:** Add SHAP or LIME explanations for individual predictions

---

## 🔌 API Integration Architecture

The project is designed with a clean separation between the ML prediction engine and a future external AI API layer.

### Current Architecture

```
User Input → Flask App → ML Model (Random Forest) → Prediction
                ↓
          services/ai_api.py (placeholder) → None (no API configured)
```

### Future Architecture (after API integration)

```
User Input → Flask App → ML Model → Prediction
                ↓                        ↓
          services/ai_api.py → External AI API → Explanation
                                                      ↓
                                              Display prediction + explanation
```

### Key Design Decisions

- The **ML model** handles all predictions. The external API will NOT replace the model.
- The external API will provide **supplementary natural-language explanations** or educational guidance.
- The application **works fully without any API key configured**.
- The `services/ai_api.py` module contains a clean interface (`get_ai_explanation()`) ready for integration.

### What I Need to Complete API Integration

To integrate the external AI API, please provide:
1. **API provider name** (e.g., OpenAI, Google Gemini, Anthropic, etc.)
2. **API endpoint URL**
3. **Authentication method** (Bearer token, API key header, etc.)
4. **Request/response format** (JSON schema or example)
5. **Your API key** (to be stored in `.env`, never committed)

---

## 🔐 API Key Security

When the API integration is added:

- API keys are stored in a `.env` file (never committed to Git)
- `.env` is listed in `.gitignore`
- A `.env.example` file is provided with placeholder values
- The application gracefully handles missing API keys
- No API key is ever placed in Python code, HTML, JavaScript, or documentation

### Setup (when ready)

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API key
# AI_API_KEY=your_actual_api_key_here
```

---

## 📁 Project Structure

```
SWYNEX-Model-or-API-Integration/
│
├── app.py                  # Flask web application
├── train_model.py          # ML training and evaluation pipeline
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── .env.example            # Environment variable template
│
├── data/
│   └── student-mat.csv     # UCI Student Performance Dataset
│
├── model/
│   └── model.pkl           # Trained model + scaler + metadata
│
├── services/
│   ├── __init__.py
│   └── ai_api.py           # AI API integration layer (placeholder)
│
├── templates/
│   └── index.html          # Web interface template
│
├── static/
│   └── style.css           # Stylesheet
│
└── results/
    ├── model_comparison.csv        # Full metrics comparison
    ├── confusion_matrices.png      # Confusion matrix plots
    └── model_comparison_chart.png  # Bar chart comparison
```

---

## 📚 Dataset Reference

P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance." In A. Brito and J. Teixeira (Eds.), Proceedings of 5th Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.

Available at: https://archive.ics.uci.edu/dataset/320/student+performance

---

*This is an educational prototype developed as part of the SWYNEX Technologies AI Internship program.*
