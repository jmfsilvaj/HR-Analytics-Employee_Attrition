# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('HR_DS.csv')
data.set_index('EmployeeNumber', inplace=True)

# Convert 'Attrition' column to binary
data['Attrition'] = data['Attrition'].map({'No': 0, 'Yes': 1})

# Label encode categorical columns
label_encoders = {}
categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 
                       'JobRole', 'MaritalStatus', 'Over18', 'OverTime']

for col in categorical_columns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    label_encoders[col] = encoder

# Split data into features and target
features = data.drop(['Attrition', 'Over18', 'StandardHours'], axis=1)
target = data['Attrition'].values

# Handle class imbalance using RandomOverSampler
oversampler = RandomOverSampler(random_state=99)
features_over, target_over = oversampler.fit_resample(features, target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_over, target_over, 
                                                    test_size=0.2, random_state=99)

# Train a RandomForest model with best parameters
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, 
                                  min_samples_split=2, min_samples_leaf=1, 
                                  bootstrap=False, random_state=99)
rf_model.fit(X_train, y_train)

# Make predictions
predictions = rf_model.predict(X_test)
