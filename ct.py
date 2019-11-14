class CleanTweet:
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self 
    def transform(self, X_train):
        """
        Args:
            X_train
        Returns:
            cleaned DataFrame
        """
        X_train = X_train.apply(lambda x: x.replace("@user", ""))
        X_train = X_train.apply(lambda x: x.replace("\n", ""))
        X_train = X_train.apply(lambda x: x.replace("&amp", ""))
        X_train = X_train.apply(lambda x: x.replace("# ", "")) # remove empty #
        X_train = X_train.apply(lambda x: x.split("@")[0])
        return X_train