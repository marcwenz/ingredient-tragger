from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from ingredient_phrase_tagger.training import cli

# TODO Fix DataLoader batch behaviour


class FaceLandmarksDataset(Dataset):
    """Ingredient tag dataset for LSTM-CRF"""

    def __init__(self):
        data = cli.Cli(100, 0).run()
        self.struct_data = self._split_data(data)

    def __len__(self):
        return len(self.struct_data)

    def __getitem__(self, idx):
        print(type(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data, features, labels, _ = self.struct_data.iloc[idx]

        return data, features, labels

    def _split_data(self, data):
        ddata = []
        for sentence, full in data:
            details = [[], [], []]
            for word in sentence:
                for ix, part in enumerate(word):
                    details[ix].append(part)
            ddata.append((*details, full))
        return pd.DataFrame(ddata)


if __name__ == "__main__":
    ds = FaceLandmarksDataset()
    print(ds[1])
