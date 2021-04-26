from flask import Flask, render_template,  redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange
from joblib import dump, load


import numpy as np  



def return_prediction(model,scaler,sample_json):
    # For larger data features, you should probably write a for loop
    # That builds out this array for you

    LotArea = sample_json['LotArea']
    YearBuilt = sample_json['YearBuilt']
    YearRemodAdd = sample_json['YearRemodAdd']
    MasVnrArea = sample_json['MasVnrArea']
    BsmtFinSF1 = sample_json['BsmtFinSF1']
    TotalBsmtSF = sample_json['TotalBsmtSF']
    dndFlrSF = sample_json['dndFlrSF']
    GrLivArea = sample_json['GrLivArea']    
    GarageArea = sample_json['GarageArea']
    WoodDeckSF = sample_json['WoodDeckSF']
    OpenPorch = sample_json['OpenPorchSF']
    MiscVal = sample_json['MiscVal']


    house = [[LotArea, YearBuilt, YearRemodAdd, MasVnrArea,BsmtFinSF1,TotalBsmtSF,dndFlrSF,GrLivArea,GarageArea,WoodDeckSF,OpenPorch,MiscVal]]
    house = scaler.transform(house)
    prediction = model.predict(house)
    
    return prediction[0]




app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'mysecretkey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
house_model = load("Ames_Housing_model(v2).h5")
house_scaler = load("Ames_Housing_scaler(v2).pkl")


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class HouseForm(FlaskForm):
    Lot_Area = TextField('Lot Area')
    Year_Built = TextField('Year Built')
    Year_RemodAdd = TextField('Year RemodAdd')
    Mas_Vnr_Area = TextField('Mas Vnr Area')
    BsmtFin_SF_1 = TextField('BsmtFin SF 1')
    Total_Bsmt_SF = TextField('Total Bsmt SF')
    dnd_Flr_SF = TextField('dnd Flr SF')
    Gr_Liv_Area = TextField('Gr Liv Area')
    Garage_Area = TextField('Garage Area')
    Wood_Deck_SF = TextField('Wood Deck SF')
    Open_Porch_SF = TextField('Open Porch SF')
    Misc_Val = TextField('Misc Val')


    submit = SubmitField('Analyze')



@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = HouseForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['Lot_Area'] = form.Lot_Area.data
        session['Year_Built'] = form.Year_Built.data
        session['Year_RemodAdd'] = form.Year_RemodAdd.data
        session['Mas_Vnr_Area'] = form.Mas_Vnr_Area.data
        session['BsmtFin_SF_1'] = form.BsmtFin_SF_1.data
        session['Total_Bsmt_SF'] = form.Total_Bsmt_SF.data
        session['dnd_Flr_SF'] = form.dnd_Flr_SF.data
        session['Gr_Liv_Area'] = form.Gr_Liv_Area.data
        session['Garage_Area'] = form.Garage_Area.data
        session['Wood_Deck_SF'] = form.Wood_Deck_SF.data
        session['Open_Porch_SF'] = form.Open_Porch_SF.data
        session['Misc_Val'] = form.Misc_Val.data

        return redirect(url_for("prediction"))


    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['LotArea'] = float(session['Lot_Area'])
    content['YearBuilt'] = float(session['Year_Built'])
    content['YearRemodAdd'] = float(session['Year_RemodAdd'])
    content['MasVnrArea'] = float(session['Mas_Vnr_Area'])
    content['BsmtFinSF1'] = float(session['BsmtFin_SF_1'])
    content['TotalBsmtSF'] = float(session['Total_Bsmt_SF'])
    content['dndFlrSF'] = float(session['dnd_Flr_SF'])
    content['GrLivArea'] = float(session['Gr_Liv_Area'])
    content['GarageArea'] = float(session['Garage_Area'])
    content['WoodDeckSF'] = float(session['Wood_Deck_SF'])
    content['OpenPorchSF'] = float(session['Open_Porch_SF'])
    content['MiscVal'] = float(session['Misc_Val'])

    results = return_prediction(model=house_model,scaler=house_scaler,sample_json=content)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)