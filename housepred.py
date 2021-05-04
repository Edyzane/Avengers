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

    L_Area = sample_json['Lot Area']
    Y_Built = sample_json['Year Built']
    Y_RemodAdd = sample_json['Year RemodAdd']
    M_Vnr_Area = sample_json['Mas Vnr Area']
    B_SF_1 = sample_json['BsmtFin SF 1']
    T_Bsmt_SF = sample_json['Total Bsmt SF']
    S_Flr_SF = sample_json['Second Flr SF']
    G_Liv_Area = sample_json['Gr Liv Area']
    G_yr_blt = sample_json['Garage Year Built']
    G_Area = sample_json['Garage Area']
    W_Deck_SF = sample_json['Wood Deck SF']
    O_Porch_SF = sample_json['Open Porch SF']
    S_porch = sample_json['screen Porch']


  

    house = [[L_Area,Y_Built, Y_RemodAdd, M_Vnr_Area,B_SF_1,T_Bsmt_SF,S_Flr_SF,G_Liv_Area,G_yr_blt,G_Area,W_Deck_SF,O_Porch_SF,S_porch]]

    house = scaler.transform(house)

    prediction = model.predict(house)

    return prediction[0]




app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'mysecretkey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
house_model = load("cs1_Ames_Housing_model.h5")
house_scaler = load("cs1_Ames_Housing_scaler.pkl")




# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class houseForm(FlaskForm):
    Lot_Area = TextField('Lot Area')
    Year_Built = TextField('Year Built')
    Year_RemodAdd = TextField('Year RemodAdd')
    Mas_Vnr_Area = TextField('Mas Vnr Area')
    BsmtFin_SF_1 = TextField('BsmtFin SF 1')
    Total_Bsmt_SF = TextField('Total Bsmt SF')
    Second_flr_SF = TextField('Second Flr SF')
    Gr_Liv_Area = TextField('Gr Liv Area')
    garage_yr_blt = TextField('Garage Year Built')
    Garage_Area = TextField('Garage Area')
    Wood_Deck_SF = TextField('Wood Deck SF')
    Open_Porch_SF = TextField('Open Porch SF')
    screen_porch = TextField('screen Porch')

    submit = SubmitField('Analyze')



@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = houseForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['Lot_Area'] = form.Lot_Area.data
        session['Year_Built'] = form.Year_Built.data
        session['Year_RemodAdd'] = form.Year_RemodAdd.data
        session['Mas_Vnr_Area'] = form.Mas_Vnr_Area.data
        session['BsmtFin_SF_1'] = form.BsmtFin_SF_1.data
        session['Total_Bsmt_SF'] = form.Total_Bsmt_SF.data
        session['Second_flr_SF'] = form.Second_flr_SF.data
        session['Gr_Liv_Area'] = form.Gr_Liv_Area.data
        session['garage_yr_blt'] = form.garage_yr_blt.data
        session['Garage_Area'] = form.Garage_Area.data
        session['Wood_Deck_SF'] = form.Wood_Deck_SF.data
        session['Open_Porch_SF'] = form.Open_Porch_SF.data
        session['screen_porch'] = form.screen_porch.data

        return redirect(url_for("prediction"))


    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['Lot Area'] = float(session['Lot_Area'])
    content['Year Built'] = float(session['Year_Built'])
    content['Year RemodAdd'] = float(session['Year_RemodAdd'])
    content['Mas Vnr Area'] = float(session['Mas_Vnr_Area'])
    content['BsmtFin SF 1'] = float(session['BsmtFin_SF_1'])
    content['Total Bsmt SF'] = float(session['Total_Bsmt_SF'])
    content['Second Flr SF'] = float(session['Second_flr_SF'])
    content['Gr Liv Area'] = float(session['Gr_Liv_Area'])
    content['Garage Year Built'] = float(session['garage_yr_blt'])
    content['Garage Area'] = float(session['Garage_Area'])
    content['Wood Deck SF'] = float(session['Wood_Deck_SF'])
    content['Open Porch SF'] = float(session['Open_Porch_SF'])
    content['screen Porch'] = float(session['screen_porch'])

    results = return_prediction(model=house_model,scaler=house_scaler,sample_json=content)

    results = np.expm1(results)
    results = "{:.2f}".format(results)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)