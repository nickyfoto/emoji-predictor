import string

def process_text(df):
    df.text = df.text.apply(lambda x: x.replace("@user", ""))
    df.text = df.text.apply(lambda x: x.replace("\n", ""))
    df.text = df.text.apply(lambda x: x.replace("&amp", ""))
    df.text = df.text.apply(lambda x: x.replace("# ", "")) # remove empty #
    df.text = df.text.apply(lambda x: x.split("@")[0])
    df.text = df.text.apply(lambda x: x.strip()) # remove empty string
    df = df[df.text != ''] # remove empty training data
    return df


class CleanTweet:
    """
    Deprecated since we have to remove 
    empty strings after preprocess
    """
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self 
    def transform(self, X_train):
        """
        Args:
            X_train, pd.Series
        Returns:
            cleaned DataFrame
        Note:
            cannot change X_train.shape
        """
        X_train = X_train.apply(lambda x: x.replace("@user", ""))
        X_train = X_train.apply(lambda x: x.replace("\n", ""))
        X_train = X_train.apply(lambda x: x.replace("&amp", ""))
        X_train = X_train.apply(lambda x: x.replace("# ", "")) # remove empty #
        X_train = X_train.apply(lambda x: x.split("@")[0])
        # X_train = X_train.apply(lambda x: x.strip()) # remove empty string
        # X_train = X_train[X_train != ''] # remove empty training data
        return X_train