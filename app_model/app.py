from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('RF_model.pk1')

# Load label encoders
label_encoders = {}
categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 
                       'JobRole', 'MaritalStatus', 'Over18', 'OverTime']

for col in categorical_columns:
    label_encoders[col] = joblib.load(f"{col}_encoder.pk1")

@app.route('/predict', methods=['GET'])
def predict():
    data = request.args.to_dict()
    df = pd.DataFrame([data])

    # Transform categorical columns using label encoders
    for col in categorical_columns:
        if col in df.columns:
            df[col] = label_encoders[col].transform(df[col])

    # Convert data types, as everything received from the URL will be a string
    for column in df.columns:
        if column not in categorical_columns:  # We've already transformed categorical columns
            df[column] = pd.to_numeric(df[column], errors='ignore')

    # Make prediction
    prediction = model.predict(df)
    
    # Return prediction
    return str(int(prediction[0]))

if __name__ == '__main__':
    app.run(port=5000, debug=True)

