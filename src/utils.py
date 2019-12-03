import pandas as pd

def get_data():
    # read in training data
    train_file = './twitter-emoji-prediction/Train.csv'
    df = pd.read_csv(train_file,
                     usecols = ['TEXT', 'Label']
                    )
    # rename columns for convenience
    df.columns = ['text', 'label']

    print('original shape', df.shape)
    df.drop_duplicates(inplace=True)
    print('after drop duplicates', df.shape)

    # read in Mapping
    mapping_file = './twitter-emoji-prediction/Mapping.csv'
    mapping = pd.read_csv(mapping_file, usecols = ['emoticons']) 
    return df, mapping

def get_top(df, mapping, n=5):
    if n == len(mapping):
        return mapping
    top = df['label'].value_counts().index[:n]
    t = mapping[mapping.index.isin(top)]
    t = t.reindex(index=top)
    return t