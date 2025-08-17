import os
import numpy as np
import pandas as pd
import joblib
import google.generativeai as genai
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS

from dotenv import load_dotenv

# --- Initialization ---
app = Flask(__name__)
load_dotenv()
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
CORS(app, origins=[frontend_url])

# Configure the Gemini AI model
try:
    genai.configure(api_key=os.getenv("API_KEY"))
    llm = genai.GenerativeModel('gemini-1.5-flash-8b')
    print("Gemini AI model configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini AI: {e}")
    llm = None

# Load the trained model and preprocessors
try:
    model = load_model('student_performance_model.keras')
    scaler = joblib.load('scaler.joblib')
    performance_encoder = joblib.load('performance_encoder.joblib')
    engagement_encoder = joblib.load('engagement_encoder.joblib')
    print("ML model and preprocessors loaded successfully.")
except Exception as e:
    print(f"Error loading model/preprocessors: {e}")
    model, scaler, performance_encoder, engagement_encoder = None, None, None, None

# --- API Endpoint ---
@app.route('/recommend', methods=['POST'])
def get_recommendation():
    if not all([model, scaler, performance_encoder, engagement_encoder, llm]):
        return jsonify({"error": "Server is not fully configured. Check logs."}), 500

    try:
        input_data = request.get_json()
        print("Received input data:", input_data)
        
        # 1. Prepare the input data for the model
        # Create a DataFrame from the input, ensuring correct column order
        features_order = [
            'Quiz_Attempts', 'Quiz_Scores', 'Forum_Participation',
            'Assignment_Completion_Rate', 'Engagement_Level', 'Final_Exam_Score',
            'Feedback_Score'
        ]
        input_df = pd.DataFrame([input_data])[features_order]

        # Encode the 'Engagement_Level' feature
        input_df['Engagement_Level'] = engagement_encoder.transform(input_df['Engagement_Level'])
        
        # 2. Scale the features using the loaded scaler
        input_scaled = scaler.transform(input_df)
        
        # 3. Predict the performance category using the Keras model
        prediction_probabilities = model.predict(input_scaled)
        predicted_index = np.argmax(prediction_probabilities, axis=1)[0]
        
        # 4. Decode the predicted index back to its string label
        performance_category = performance_encoder.inverse_transform([predicted_index])[0]

        # 5. Generate prompt for the LLM
        prompt = f"""
            Act as an expert academic advisor. A student's performance has been predicted as "{performance_category}" based on the following metrics:
            - Final Exam Score: {input_data['Final_Exam_Score']}%
            - Average Quiz Score: {input_data['Quiz_Scores']}%
            - Assignment Completion Rate: {input_data['Assignment_Completion_Rate']}%
            - Forum Participation: {input_data['Forum_Participation']} posts
            - Engagement Level: {input_data['Engagement_Level']}
            - Quiz Attempts: {input_data['Quiz_Attempts']}
            - Feedback Score: {input_data['Feedback_Score']} out of 5

            Based on their "{performance_category}" performance category and their specific metrics, provide a set of personalized, actionable recommendations. Structure your response in Markdown format.

            Your response should include:
            1.  A brief, encouraging summary of their performance category.
            2.  Specific recommendations for learning materials (e.g., suggest types of online courses, specific YouTube channels, relevant articles, or books).
            3.  Actionable study strategies tailored to their situation.
            4.  Suggestions for how to improve or leverage their engagement and participation.

            Tailor the tone and content appropriately for the "{performance_category}" category.
            """

        # 6. Get recommendation from Gemini AI
        response = llm.generate_content(prompt)
        
        return jsonify({
            "predicted_performance_category": performance_category,
            "recommendation": response.text
        })
    

    except Exception as e:
        # Provide a more specific error message if a key is missing
        if isinstance(e, KeyError):
            return jsonify({"error": f"Missing required input feature: {e}"}), 400
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
