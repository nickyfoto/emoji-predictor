install python-3.7.3


pyenv install 3.7.3


## run flask app locally

export FLASK_APP=predictor_app.py
pip3 install -r requirements.txt
python3 -m flask run --host=0.0.0.0


sudo snap install heroku --classic
heroku create emoji-predictor

heroku git:remote -a emoji-predictor
git push heroku master


url

https://emoji-predictor.herokuapp.com/



# Twitch Harassment Classifier Website
Showcases harassment classifier trained on 160,000+ Wikipedia comments from a Kaggle dataset. Primary goal is to pipe live Twitch chat from a streaming channel and classify comments for toxicity in real time.

Model is currently live and available on this [Twitch channel](https://www.twitch.tv/datatestdummy/). To see predictions, type toxic chat into the chatbar on the right of the screen. Note: Website may take 10 seconds to load if its heroku node is currently sleeping due to inactivity.

By Jeremy Chow and Randy Macaraeg
