# 🌳 Decision Tree & Random Forest Classifier

## 📌 Overview
This project demonstrates end-to-end training, visualization, and evaluation of **Decision Tree** and **Random Forest** classifiers using the heart.csv dataset.  
It includes:
1. Training a **Decision Tree Classifier**.
2. Plotting the decision tree for interpretability.
3. Detecting and reducing **overfitting** using `max_depth`.
4. Training a **Random Forest Classifier** for comparison.
5. Performing **feature importance analysis**.
6. Evaluating models using **cross-validation**.

---

## 📊 Dataset
- **Source:** [https://www.kaggle.com/datasets/arezaei81/heartcsv]

---

## 🚀 Results Summary

| Model                               | Accuracy |
|-------------------------------------|----------|
| Decision Tree (default)             | 0.7541   |
| Decision Tree (max_depth = 4, train)| 0.8843   |
| Decision Tree (max_depth = 4, test) | 0.8525   |
| Random Forest                       | 0.8361   |

---

## 📈 Key Observations
- The **default Decision Tree** was prone to overfitting, performing well on training data but worse on test data.
- Restricting the tree depth (`max_depth = 4`) improved generalization.
- **Random Forest** provided competitive accuracy with more stability.
- Feature importance analysis revealed which features drive classification decisions the most.

---

## 📷 Visualizations
The project generates:
1. **Decision Tree Plot** – showing splitting rules and thresholds.
2. **Feature Importance Bar Chart** – showing the contribution of each feature.



