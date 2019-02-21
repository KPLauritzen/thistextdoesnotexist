# thistextdoesnotexist
www.thistextdoesnotexist.com

## Initial plan overview
- Make a Flask app. This way I don't need to know any html
- Figure out how the scripts provided by OpenAI works
- Make a FlaskForm that calls the RNN function 
- Call function with a text prompt
- If this works locally, put it on Google App Engine
- Figure out what Google App Engine costs
- Point the newly online app to our domain
- ???
- PROFIT
## Usage
1) Get required code
```
git clone https://github.com/KPLauritzen/thistextdoesnotexist.git thistext
cd thistext
pip install -r requirements.txt
```
2) run Flask app
```
python app/app.py
```

This will take a while the first time, as it has to download the pretrained model.

3) See if it works:

Open your browser at `localhost:5000`
Input some weird stuff and see how the model reacts. This is very slow, expect 30s wait.
