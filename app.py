#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
from flask import request, render_template
from keras.models import load_model

import pandas as pd
import pickle


# In[4]:


app = Flask(__name__)


# In[ ]:


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method=="POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        
        df = pd.DataFrame(columns=['income', 'age', 'loan'])
        df = df.append({'income': float(income), 'age': int(age), 'loan': float(loan)}, ignore_index=True)
        
        import joblib
        scaler = joblib.load("MinMaxScaler")
        scaled = scaler.transform([df.iloc[0]])

        print(income, age, loan)
        print(scaled)
        from keras.models import load_model

        logistic_model = joblib.load("logistic_reg")
        Dtree_model = joblib.load("decision_tree")
        RF_model = joblib.load("random_forest")
        xgb_model = joblib.load('xgboost')
        NN_model = load_model('neural_net')
        
        logistic_pred = logistic_model.predict([[float(income), int(age), float(loan)]])
        Dtree_pred = Dtree_model.predict([[float(income), int(age), float(loan)]])
        RF_pred = RF_model.predict([[float(income), int(age), float(loan)]])
        xgb_pred = xgb_model.predict(df)
        NN_pred = NN_model.predict([scaled])
        
        print(logistic_pred)
        print(Dtree_pred)
        print(RF_pred)
        print(xgb_pred)
        print(NN_pred)
        
        if logistic_pred[0] == 0:
            s1 = 'You will not default [Logistic model]'
        else:
            s1 = 'You will default [Logistic model]'
            
        if Dtree_pred[0] == 0:
            s2 = 'You will not default [Decision tree model]'
        else:
            s2 = 'You will default [Decision tree model]'
        
        if RF_pred[0] == 0:
            s3 = 'You will not default [Random Forest model]'
        else:
            s3 = 'You will default [Random Forest model]'
        
        if xgb_pred[0] == 0:
            s4 = 'You will not default [XGBoost model]'
        else:
            s4 = 'You will default [XGBoost model]'
            
        if NN_pred[0][0] < 0.5:
            s5 = 'You will not default with default probability of only' + ' ' + str(NN_pred[0][0]) + '[NN model]'
        else:
            s5 = 'You will default with default probability of' + ' ' + str(NN_pred[0][0]) + '[NN model]'
            
        return(render_template("index.html", result1 = s1, result2 = s2, result3 = s3, result4 = s4, result5 = s5))
    
    else:
        return(render_template("index.html", result1='', result2='', result3='', result4='', result5=''))


# In[ ]:


if __name__ =="__main__":
     app.run()


# In[ ]:




