import numpy as np
import pandas as pd
import joblib 
from custom_transformers import column_names, yeojohnson, boxcox

loaded_model = joblib.load('rfc_model_final.joblib')

input_data = pd.DataFrame([{'Age':25,'Gender':1,'City':'C20',
                            'Education_Level':3,'Income':25690,'Joining Designation':1,
                            'Grade':2,'Total Business Value':940839,'Quarterly Rating':3,
                            'income_increased':0,'rating_increased':1}])

print(loaded_model.predict(input_data))