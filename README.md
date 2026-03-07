# Car Evaluation Classification — Decision Tree vs Random Forest (Custom + sklearn)
This notebook trains and evaluates three tree-based models on the UCI Car Evaluation dataset:
1.	Decision Tree (sklearn)
2.	Random Forest (sklearn)
3.	MyRandomForest (custom implementation) — built using:
	-	bootstrapping (sampling with replacement)
	- multiple DecisionTreeClassifiers
	- majority-vote aggregation

The notebook includes preprocessing, model training, evaluation metrics, and confusion matrices.


## Dataset
- Dataset: Car Evaluation (UCI)
- Source: Loaded using ucimlrepo (dataset id = 19)
- Rows: 1,728
- Features: 6 categorical features (buying, maint, doors, persons, lug_boot, safety)
- Target: class (4 classes: unacc, acc, good, vgood)


## What the Notebook Does
1. Load Data
	- Fetches dataset using fetch_ucirepo(id=19)
	- Separates features X and target y
	- Plots class distribution (bar chart)

2. Preprocessing
	- One-hot encodes all categorical features using pd.get_dummies()
	- Splits into train/test with stratified sampling (80/20)

3. Train Models
	- Trains a DecisionTreeClassifier
	- Trains sklearn’s RandomForestClassifier
	- Trains MyRandomForest (custom)

4. Evaluate
For each model, the notebook prints:
	- Accuracy
	- Classification report (precision/recall/F1)
	- Confusion matrix plot

Finally, it prints a comparison table:
	- Model name
	- Accuracy


## Custom Model: MyRandomForest
MyRandomForest class works like a simplified Random Forest:
	- Bootstrapping: for each tree, sample n rows with replacement
	- Train a DecisionTreeClassifier on each bootstrap sample
	- Prediction: collect predictions from all trees and return the majority vote
	- Also includes predict_proba() by averaging tree probabilities


## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn
