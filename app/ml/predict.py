import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any
import os


class TravelTimePredictor:
    """
    Handles loading the ML model and making travel time predictions.
    """

    def __init__(self):
        """Initialize predictor by loading model and encoders."""
        self.model = None
        self.label_encoders = None
        self.feature_names = None
        self.model_loaded = False

    def load_model(self):
        """
        Load the trained XGBoost model and label encoders from disk.
        This is called once when the API starts.
        """
        try:
            # Load the trained XGBoost model
            with open('models/xgboost_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print("✅ XGBoost model loaded successfully")

            # Load label encoders for categorical features
            with open('models/label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            print("✅ Label encoders loaded successfully")

            # Load feature names (column order)
            with open('models/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            print("✅ Feature names loaded successfully")

            self.model_loaded = True
            print("🎉 Prediction service ready!")

        except FileNotFoundError as e:
            print(f"❌ Error: Model files not found. Please train the model first.")
            print(f"   Run: python app/ml/train.py")
            raise e
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data to match the format expected by the model.

        Args:
            input_data: Dictionary with keys:
                - distance_km
                - traffic_hours
                - vehicle_avg_speed
                - vehicle_type
                - road_condition
                - weather_condition

        Returns:
            DataFrame with encoded features ready for prediction
        """
        # Create a copy to avoid modifying original
        processed = input_data.copy()

        # Encode categorical features using saved label encoders
        categorical_features = ['vehicle_type', 'road_condition', 'weather_condition']

        for feature in categorical_features:
            if feature in processed:
                # Get the label encoder for this feature
                encoder = self.label_encoders[feature]

                # Transform the categorical value to encoded number
                # e.g., "car" -> 1, "rain" -> 4
                encoded_value = encoder.transform([processed[feature]])[0]

                # Add encoded column
                processed[feature + '_encoded'] = encoded_value

        # Create DataFrame with features in correct order
        feature_dict = {
            'distance_km': processed['distance_km'],
            'traffic_hours': processed['traffic_hours'],
            'vehicle_avg_speed': processed['vehicle_avg_speed'],
            'vehicle_type_encoded': processed['vehicle_type_encoded'],
            'road_condition_encoded': processed['road_condition_encoded'],
            'weather_condition_encoded': processed['weather_condition_encoded']
        }

        # Convert to DataFrame (model expects DataFrame input)
        df = pd.DataFrame([feature_dict])

        # Ensure columns are in the same order as training
        df = df[self.feature_names]

        return df

    def predict(self, input_data: Dict[str, Any]) -> float:
        """
        Make a travel time prediction.

        Args:
            input_data: Dictionary with trip features

        Returns:
            Predicted travel time in minutes
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess input (encode categories, create DataFrame)
        processed_data = self.preprocess_input(input_data)

        # Make prediction using the trained model
        prediction = self.model.predict(processed_data)[0]

        # Round to 2 decimal places
        prediction = round(float(prediction), 2)

        return prediction

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        if not self.model_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_type": "XGBoost Regressor",
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "categorical_encoders": list(self.label_encoders.keys())
        }


# Create a global instance (singleton pattern)
# This is loaded once when the API starts and reused for all requests
predictor = TravelTimePredictor()
