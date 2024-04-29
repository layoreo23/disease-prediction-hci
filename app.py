from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

symptoms = ['itching', 'continuous_sneezing', 'shivering', 'joint_pain',
       'stomach_pain', 'vomiting', 'fatigue', 'weight_loss', 'restlessness',
       'lethargy', 'high_fever', 'headache', 'dark_urine', 'nausea',
       'pain_behind_the_eyes', 'constipation', 'abdominal_pain', 'diarrhoea',
       'mild_fever', 'yellowing_of_eyes', 'malaise', 'phlegm', 'congestion',
       'chest_pain', 'fast_heart_rate', 'neck_pain', 'dizziness',
       'puffy_face_and_eyes', 'knee_pain', 'muscle_weakness',
       'passage_of_gases', 'irritability', 'muscle_pain', 'belly_pain',
       'abnormal_menstruation', 'increased_appetite', 'lack_of_concentration',
       'visual_disturbances', 'receiving_blood_transfusion', 'coma',
       'history_of_alcohol_consumption', 'blood_in_sputum', 'palpitations',
       'inflammatory_nails', 'yellow_crust_ooze']

@app.route("/")
def home():
    return render_template('index.html', word="Home")

@app.route('/details')
def pred():
    return render_template('details.html', word="Details", symptoms=symptoms)

@app.route('/predict',methods=['POST','GET'])
def predict():
    col=['itching', 'continuous_sneezing', 'shivering', 'joint_pain',
       'stomach_pain', 'vomiting', 'fatigue', 'weight_loss', 'restlessness',
       'lethargy', 'high_fever', 'headache', 'dark_urine', 'nausea',
       'pain_behind_the_eyes', 'constipation', 'abdominal_pain', 'diarrhoea',
       'mild_fever', 'yellowing_of_eyes', 'malaise', 'phlegm', 'congestion',
       'chest_pain', 'fast_heart_rate', 'neck_pain', 'dizziness',
       'puffy_face_and_eyes', 'knee_pain', 'muscle_weakness',
       'passage_of_gases', 'irritability', 'muscle_pain', 'belly_pain',
       'abnormal_menstruation', 'increased_appetite', 'lack_of_concentration',
       'visual_disturbances', 'receiving_blood_transfusion', 'coma',
       'history_of_alcohol_consumption', 'blood_in_sputum', 'palpitations',
       'inflammatory_nails', 'yellow_crust_ooze']
    if request.method=='POST':
        inputt = [str(x) for x in request.form.values()]

        b=[0]*45
        for x in range(0,45):
            for y in inputt:
                if(col[x]==y):
                    if y in ["stomach_pain", "vomiting", "nausea", "abdominal_pain", "diarrhoea"]:
                            prediction = "Gastroentritis"
                            return render_template('results.html', word="Prediction", prediction_text="The probable diagnosis says it could be {}. For more information: <a href='https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7119329/'>Click here</a>".format(prediction))
                    if y in ["increased_appetite", "continuous_sneezing", "restlessness", "lethargy", "high_fever", "fatigue"]:
                            prediction = "Common Cold/Infection"
                            return render_template('results.html', word="Prediction", prediction_text="The probable diagnosis says it could be {}. For more information: <a href='https://www.cdc.gov/antibiotic-use/colds.html'>Click here</a>".format(prediction))
                    b[x]=1
                    
        b=np.array(b)
        b=b.reshape(1,45)
        prediction = model.predict(b)
        prediction = prediction[0]
    return render_template('results.html', word="Prediction", prediction_text="The probable diagnosis says it could be symtoms of {}.Please visit a General Doctor for further advice.".format(prediction))


if __name__ == "__main__":
    app.run()