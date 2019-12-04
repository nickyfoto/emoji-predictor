# Emoji Predictor

deploy [premoji](https://github.com/nickyfoto/premoji/) on heroku.

## Development

### run flask app locally

```
export FLASK_APP=predictor_app.py
pip3 install -r requirements.txt
python3 -m flask run --host=0.0.0.0
```

### Deploy to heroku

- install heroku cli, in ubuntu

	```
	sudo snap install heroku --classic
	```

- create a heroku app
	
	```
	heroku create <app-name>
	```

- add heroku as a remote repo

	```
	heroku git:remote -a emoji-predictor
	git push heroku master
	```
## Todos

## Credit

Heroku Deploy template from [Twitch Harassment Classifier Website](https://github.com/jeremyrchow/Harassment-Classifier-App)
