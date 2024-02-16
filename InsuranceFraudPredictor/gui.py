import streamlit as st
import pandas as pd
import pickle

# Load the saved model
model = pickle.load(open('Fraud_Predictor.pkl', 'rb'))

# Create a form for users to input new data
st.title('Insurance Fraud Predictor')
form = st.form(key='input_form')
age = form.number_input('Age', value=18)
policy_csl = form.selectbox('Policy Csl',['100/300' ,'250/500' ,'500/1000'])
policy_deductable = form.number_input('Policy Deductable', value=1)
policy_annual_premium = form.number_input('Policy Annual Premium', value=1.0,step=0.01)
umbrella_limit = form.number_input('Umbrella Limit', value=1)
insured_sex = form.selectbox('Insured Sex', ['Male', 'Female'])
insured_education_level = form.selectbox('Insured Education Level',
                                         ['JD', 'High School', 'MD', 'Associate', 'Masters', 'PhD'])
insured_occupation = form.selectbox('Insured Occupation',
                                    ['craft-repair', 'machine-op-inspct', 'sales', 'armed-forces', 'tech-support',
                                     'exec-managerial', 'prof-specialty', 'other-service', 'priv-house-serv',
                                     'transport-moving', 'handlers-cleaners', 'adm-clerical', 'farming-fishing',
                                     'protective-serv'])
insured_relationship = form.selectbox('Insured Relationship',
                                      ['other-relative', 'not-in-family', 'husband', 'wife', 'own-child',
                                       'unmarried'])
capital_gains = form.number_input('Capital Gains', value=1)
capital_loss = form.number_input('Capital Loss', value=1)
incident_type = form.selectbox('Incident Type',
                               ['Multi-vehicle Collision', 'Single Vehicle Collision', 'Parked Car'])
collision_type = form.selectbox('Collision Type', ['Front Collision', 'Rear Collision', 'Side Collision'])
incident_severity = form.selectbox('Incident Severity',
                                   ['Major Damage', 'Minor Damage', 'Total Loss', 'Trivial Damage'])
authorities_contacted = form.selectbox('Authorities Contacted', ['Police', 'Fire', 'Other', 'None', 'Ambulance'])
incident_hour_of_the_day = form.number_input('Incident Hour of the Day', value=1)
number_of_vehicles_involved = form.number_input('Number of Vehicles Involved', value=1)
property_damage = form.selectbox('Property Damage', ['NO', 'YES'])
bodily_injuries = form.number_input('Bodily Injuries', value=1)
witnesses = form.number_input('Witnesses', value=1)
police_report_available = form.selectbox('Police Report Available', ['NO', 'YES'])
injury_claim = form.number_input('Injury Claim', value=1)
property_claim = form.number_input('Property Claim', value=1)
vehicle_claim = form.number_input('Vehicle Claim', value=1)
form_submit = form.form_submit_button(label='Predict')

if form_submit:
    # preprocess the input data
    data = pd.DataFrame({
        'age': [age],
        'policy_csl' : [policy_csl],
        'policy_deductable': [policy_deductable],
        'policy_annual_premium': [policy_annual_premium],
        'umbrella_limit': [umbrella_limit],
        'insured_sex': [insured_sex],
        'insured_education_level': [insured_education_level],
        'insured_occupation': [insured_occupation],
        'insured_relationship': [insured_relationship],
        'capital_gains':[capital_gains],
        'capital_loss': [capital_loss],
        'incident_type': [incident_type],
        'collision_type': [collision_type],
        'incident_severity': [incident_severity],
        'authorities_contacted': [authorities_contacted],
        'incident_hour_of_the_day': [incident_hour_of_the_day],
        'number_of_vehicles_involved': [number_of_vehicles_involved],
        'property_damage': [property_damage],
        'bodily_injuries': [bodily_injuries],
        'witnesses': [witnesses],
        'police_report_available': [police_report_available],
        'injury_claim': [injury_claim],
        'property_claim': [property_claim],
        'vehicle_claim': [vehicle_claim]
    })

    # make a prediction using the loaded model
    prediction = model.predict(data)[0]

    # display the prediction result to the user
    if prediction == 0:
       st.write('NOT A FRAUD')
    else:
       st.write('FRAUD')



