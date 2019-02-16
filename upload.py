# import all the dependies 
import os
from flask import Flask, render_template, request, jsonify
from werkzeug import secure_filename
from flask_wtf import FlaskForm, Form
from flask_wtf.file import FileField
from wtforms import SubmitField

import json
import numpy as np
from PIL import Image
from flask_bootstrap import Bootstrap #this make our final web application look more Beautiful 


import keras as k #import Keras 
from keras.models import Sequential 
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array

#Path defined to save uploaded Images by the user from web app
UPLOAD_FOLDER = 'YOUR/PATH/TO/SAVE/images'

#function to load the
def get_model():
	global model 
	model = load_model("Amy_or_Isla.h5")
	print(" * Model is Loaded BOOS!")
print(" * Model Is loading.....")
get_model()

#init the Flask 'app' 
app = Flask(__name__, static_folder="images")
Bootstrap(app)

#Configure the upload folder that we defiend 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#create a class that inherits from 'Form' that we instanthiat in our "@app.route"
class upload_form(Form):
    photo = FileField("Your Photo")
    submit = SubmitField(u'Upload')



#define route at "/"
@app.route('/', methods=['GET','POST'])
#The upload_file() function will be called at "/" route
def upload_file():
    #Init our form class that we defied up
    form = upload_form(csrf_enabled=False)

    #When the user will upload the file and the click "upload button" this if condition will be true
    if request.method == 'POST' and form.validate():
        #get the uploaded Image, and Image name by the user
        file = request.files['photo']
        filename = file.filename
        #save image
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        source = "C:/Users/Parth/Desktop/Prototype/images/" + filename

        #open and resize Image in 224 X 244
        Img = Image.open(source)
        Img = Img.resize((224, 224))
        Img.save("Your/Path/here/pred.jpg", 'JPEG')

        #Convert Image into a form that can be used as input to the model
        img = image.load_img('Your/Path/here/pred.jpg')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        #model will make prediction here 
        prediction = model.predict(x).tolist()

        #prediction --> as Dict.
        responce = {
    		'prediction' : {
    			'Amy' : prediction[0][0],
    			'Isla' : prediction[0][1]
    		}
    	}
        ans = responce

    else:
        ans = None
        filename = "Q.jpg"

    #rendring the templet
    return render_template('index.html', form=form, ans=json.dumps(ans, sort_keys = False, indent = 2),Disp_Img = filename)

#Run when it's called 
if __name__ == '__main__':
    app.run()