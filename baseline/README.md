# Baseline Model #

Baseline model which tokenizes Java code, 
vectorizes corpus with TF/IDF vectorizer and builds 
a Random Forest classifier from the features.

## Running
* Install requirements: `pip install -r requirements.txt`  
* Tokenize code and fit the model: `sh preprocess.sh && python model.py`  