from flask import Flask, render_template,  redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange
from joblib import dump, load
import numpy as np


import numpy as np  



def return_prediction(model,scaler,sample_json):
    # For larger data features, you should probably write a for loop
    # That builds out this array for you

    age= sample_json['age']
    sex = sample_json['sex']
    cp = sample_json['cp']
    trestbps = sample_json['trestbps']
    chol = sample_json['chol']
    fbs = sample_json['fbs']
    restecg = sample_json['restecg']
    thalach = sample_json['thalach']
    exang = sample_json['exang']
    oldpeak = sample_json['oldpeak']
    slope = sample_json['slope']
    ca = sample_json['ca']
    thal = sample_json['thal']



  

    heart = [[age,sex,cp, trestbps, chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]

    heart = scaler.transform(heart)

    prediction = model.predict(heart)

    if prediction == 1:
            res = 'Affected'
    else:
        res = 'Not affected'

    #return prediction[0]
    return res



app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'mysecretkey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
house_model = load("heart_model.h5")
house_scaler = load("heart_scaler.pkl")




# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class houseForm(FlaskForm):
    age = TextField('age')
    sex = TextField('sex')
    cp = TextField('cp')
    trestbps = TextField('trestbps')
    chol = TextField('chol')
    fbs = TextField('fbs')
    restecg = TextField('restecg')
    thalach = TextField('thalach')
    exang = TextField('exang')
    oldpeak = TextField('oldpeak')
    slope = TextField('slope')
    ca = TextField('ca')
    thal = TextField('thal')
   


    submit = SubmitField('Analyze')



@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = houseForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['age'] = form.age.data
        session['sex'] = form.sex.data
        session['cp'] = form.cp.data
        session['trestbps'] = form.trestbps.data
        session['chol'] = form.chol.data
        session['fbs'] = form.fbs.data
        session['restecg'] = form.restecg.data
        session['thalach'] = form.thalach.data
        session['exang'] = form.exang.data
        session['oldpeak'] = form.oldpeak.data
        session['slope'] = form.slope.data
        session['ca'] = form.ca.data
        session['thal'] = form.thal.data

        return redirect(url_for("prediction"))


    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['age'] = float(session['age'])
    content['sex'] = float(session['sex'])
    content['cp'] = float(session['cp'])
    content['trestbps'] = float(session['trestbps'])
    content['chol'] = float(session['chol'])
    content['fbs'] = float(session['fbs'])
    content['restecg'] = float(session['restecg'])
    content['thalach'] = float(session['thalach'])
    content['exang'] = float(session['exang'])
    content['oldpeak'] = float(session['oldpeak'])
    content['slope'] = float(session['slope'])
    content['ca'] = float(session['ca'])
    content['thal'] = float(session['thal'])

    results = return_prediction(model=house_model,scaler=house_scaler,sample_json=content)

    #results = np.expm1(results)
    #results = "{:.2f}".format(results)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)