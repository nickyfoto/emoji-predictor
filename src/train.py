"""
train and save model

20 emojis

"""
from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

from utils import get_data, get_top
from ct import CleanTweet, process_text

df, mapping = get_data()

emojis = get_top(df, mapping, n=len(mapping))
df = process_text(df)
data = df


X_train, X_test, y_train, y_test = train_test_split(data.text, data.label, test_size=0.1, random_state=42, 
                                                    stratify=data.label)


# ct = CleanTweet()
# X_train = ct.transform(X_train)


vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english')
vectorized_X_train = vectorizer.fit_transform(X_train)

# dump(vectorizer, 'input_vectorizer.joblib')

text_clf = Pipeline([
    # ('clean', CleanTweet()),
    ('vect', TfidfVectorizer(strip_accents='ascii', stop_words='english')),
    # ('clf', MultinomialNB()),
    ('clf', LogisticRegression(
    					  # penalty='elasticnet',
    					  random_state=0, 
    					  # solver = 'saga',
    					  # l1_ratio = 0.5,
    					  # class_weight = 'balanced',
    					  solver='lbfgs',
                          # multi_class='multinomial',
                          multi_class='ovr',
                          max_iter=10000, 
                          verbose=1, 
                          n_jobs=-1,
                          tol=1e-6))
    ])
text_clf.fit(X_train, y_train)

training_score = text_clf.score(X_train, y_train)
print("Train Accuracy:", training_score)
testing_score = text_clf.score(X_test, y_test)
print("Test Accuracy:", testing_score)

dump(text_clf, 'emojis_model_19120301.joblib')