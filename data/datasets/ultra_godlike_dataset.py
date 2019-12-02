import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data.datasets.pets_dataset import PetsDataset


class UltraGodlikeDataset(Dataset):
    def __init__(self, train_imgs_dir, csv_path, label_column, is_train, categorical_fields):
        super(UltraGodlikeDataset, self).__init__()
        self.imgs_dataset = PetsDataset(train_imgs_dir, csv_path, label_column, is_train)
        self.cat_fields = categorical_fields
        df = self.preprocess(pd.read_csv(csv_path, dtype={"PetID": str}))
        df = self.filter_train_test(df, is_train)
        EmbeddingDataPreprocess(df, self.cat_fields)

        self.embedding_dataset = EmbeddingDataset.from_data_frame(df, self.cat_fields)

    def __getitem__(self, index):
        return (self.imgs_dataset.__getitem__(index)[0], self.embedding_dataset.__getitem__(index)[0],
                self.embedding_dataset.__getitem__(index)[1]), \
               self.imgs_dataset.__getitem__(index)[1]

    def __len__(self):
        return len(self.imgs_dataset)

    def filter_train_test(self, all_data_info, is_train):
        set_id = 0 if is_train else 1
        set_idxs = all_data_info["set"] == set_id
        df = all_data_info.loc[set_idxs]
        df = df.loc[(df['Quantity'] == 1) & (df['PhotoAmt'] > 0)]
        df = df.drop(['set','Quantity'], axis=1)

        return df

    def preprocess(self, A):
        A['Type'] = A['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')

        # binary noname col
        A['Name'] = A['Name'].fillna('Unnamed')
        A['No_name'] = 0
        A.loc[A['Name'] == 'Unnamed', 'No_name'] = 1

        A['binned_age'] = pd.cut(A['Age'], bins=[-1, 3, 6, 12, 24, 36, 10000])

        # Meaningless names - 2 characters and less (maybe 3 as well)
        # A['meaningless_name'] = 0
        # A.loc[len(A['Name']) <= 2, 'meaningless_name'] = 1

        # is pure-bred
        A['Pure_breed'] = 0
        A.loc[A['Breed2'] == 0, 'Pure_breed'] = 1

        A['health'] = A['Vaccinated'].astype(str) + '_' + \
                      A['Dewormed'].astype(str) + '_' + \
                      A['Sterilized'].astype(str) + '_' + \
                      A['Health'].astype(str)

        A['Free'] = A['Fee'].apply(lambda x: 1 if x == 0 else 0)

        A['Description'] = A['Description'].fillna('')
        A['desc_length'] = A['Description'].apply(lambda x: len(x))
        A['desc_words'] = A['Description'].apply(lambda x: len(x.split()))
        A['averate_word_length'] = A['desc_length'] / A['desc_words']
        A.loc[~np.isfinite(A['averate_word_length']), 'averate_word_length'] = 0

        return A.drop(columns=['Unnamed: 0'] + ['Description', 'Name', 'PetID', 'RescuerID', 'health'] + ['BinaryLabel',
                                                                                                          'AdoptionSpeed',
                                                                                                          'binned_age'
                                                                                                          ])


def EmbeddingDataPreprocess(data, cats, inplace=True):
    ### Each categorical column should have indices as values
    ### Which will be looked up at embedding matrix and used in modeling
    ### Make changes inplace
    if inplace:
        for c in cats:
            data[c].replace({val: i for i, val in enumerate(data[c].unique())}, inplace=True)
        return data
    else:
        data_copy = data.copy()
        for c in cats:
            data_copy[c].replace({val: i for i, val in enumerate(data_copy[c].unique())}, inplace=True)
        return data_copy


class EmbeddingDataset(Dataset):
    ### This dataset will prepare inputs cats, conts and output y
    ### To be feed into our mixed input embedding fully connected NN model
    ### Stacks numpy arrays to create nxm matrices where n = rows, m = columns
    ### Gives y 0 if not specified
    def __init__(self, cats, conts, y):
        n = len(cats[0]) if cats else len(conts[0])
        self.cats = np.stack(cats, 1).astype(np.int64) if cats else np.zeros((n, 1))
        self.conts = np.stack(conts, 1).astype(np.float32) if conts else np.zeros((n, 1))
        self.y = np.zeros((n, 1)) if y is None else y[:, None].astype(np.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx], self.y[idx]]

    @classmethod
    def from_data_frames(cls, df_cat, df_cont, y=None):
        cat_cols = [c.values for n, c in df_cat.items()]
        cont_cols = [c.values for n, c in df_cont.items()]
        return cls(cat_cols, cont_cols, y)

    @classmethod
    def from_data_frame(cls, df, cat_flds, y=None):
        return cls.from_data_frames(df[cat_flds], df.drop(cat_flds, axis=1), y)

    ### We will keep this for fastai compatibility
