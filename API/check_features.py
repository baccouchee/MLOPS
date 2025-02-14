import joblib

# Load the model
model = joblib.load('/app/model/random_forest_model.pkl')

# Print the feature names
print("Feature names used during model training:")
print(model.feature_names_in_)