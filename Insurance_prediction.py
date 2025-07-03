import pickle
import numpy as np
import streamlit as st

# Load the trained model and scaler
loaded_model = pickle.load(open('training_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Prediction function
def premium_prediction(input_data):
    # Transform input data to numpy array
    input_data_as_numpy_array = np.array(input_data).reshape(1, -1)
    
    # Predict the premium
    prediction = loaded_model.predict(scaler.transform(input_data_as_numpy_array))
    return prediction[0]

# Streamlit UI
def main():
    st.title("💰 Health Insurance Premium Prediction Web App")

    # Input fields for user
    age = st.number_input('Age of the Person')
    sex = st.selectbox('Gender', ['Male', 'Female'])
    bmi = st.number_input('BMI Value')
    children = st.number_input('Number of Children')
    smoker = st.selectbox('Is the person a Smoker?', ['Yes', 'No'])
    region = st.selectbox('Region', ['Northeast', 'Northwest', 'Southeast', 'Southwest'])

    # Convert categorical inputs to numerical values
    sex = 1 if sex == 'Male' else 0
    smoker = 1 if smoker == 'Yes' else 0
    region_mapping = {'Northeast': 0, 'Northwest': 1, 'Southeast': 2, 'Southwest': 3}
    region = region_mapping.get(region, 0)  # Default to Northeast if not found

    # Button to calculate premium
    if st.button('Calculate Premium'):
        # Prepare input data for prediction
        input_data = [age, sex, bmi, children, smoker, region]

        # Call prediction function
        premium = premium_prediction(input_data)

        # Display the prediction result
        st.success(f'The Health Insurance Premium is: ${premium:.2f}')

# Run the app
if __name__ == '__main__':
    main()
