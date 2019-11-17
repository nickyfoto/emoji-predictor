"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.

This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

# import pickle
import numpy as np
# from sklearn.externals import joblib
import joblib
# import re
import pandas as pd
# Load the models 
# model_dict is the collection of extra tree models 

# This line doesn't work, joblib only loads locally. File is too big to upload to heroku though
# model_dict = joblib.load('https://drive.google.com/open?id=1h20N5Cooti2e5CDkmKY5LOzRuLksyR5e')
# model_dict = joblib.load('./static/models/models_compressed.p')
# word_vectorizer = joblib.load('static/models/word_vectorizer.p')

# model_dict = joblib.load('./static/models/log_models.p')
# model = joblib.load('./static/models/emojis_model.joblib')
model = joblib.load('./static/models/emojis_model_19111601.joblib')
mapping_file = './static/models/Mapping.csv'
emojis = pd.read_csv(mapping_file, usecols = ['emoticons']) 

def make_prediction(input_chat):
    """
    Given string to classify, returns the input argument and the dictionary of 
    model classifications in a dict so that it may be passed back to the HTML page.

    Input:
    Raw string input

    Function makes sure the features are fed to the model in the same order the
    model expects them.

    Output:
    Returns (x_inputs, probs) where
      x_inputs: a list of feature values in the order they appear in the model
      probs: a list of dictionaries with keys 'name', 'prob'
    """

    if not input_chat:
        input_chat = ' '
    if len(input_chat) > 500:
        input_chat = input_chat[:500]
    input_df = pd.DataFrame({'text': [input_chat]})
    # print(input_df)
    probs = model.predict_proba(input_df.text)


    top_5_indices = np.argsort(probs)[0][::-1][:5]
    # print(top_5_indices)
    # predictions = [emojis.emoticons[i] for i in top_5_indices]
    predictions = [{'name': emojis.emoticons[i], 'prob': probs[0][i]} for i in top_5_indices]
    # print(predictions)
    return (input_chat, predictions)

# This section checks that the prediction code runs properly
# To test, use "python predictor_api.py" in the terminal.

# if __name__='__main__' section only runs
# when running this file; it doesn't run when importing

if __name__ == '__main__':
    from pprint import pprint
    # print("Checking to see what empty string predicts")
    # print('input string is ')
    def test_key_words(chat_in, note):
        x_input, probs = make_prediction(chat_in)
        print(f'Input values: {x_input}, note: {note}')
        pprint(probs)
    
    test_key_words('usa', 'us flag')
    test_key_words('tree', 'tree')
    test_key_words('christmas', 'tree')
    test_key_words('camera', 'camera')
    test_key_words('photo', '')
    test_key_words('cry', 'cry')