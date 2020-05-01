from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from ingredient_phrase_tagger.training import cli


def collate(batch_data):
    """
    Returns batched data in form
    [
        [
            *len(batch_data) lists of sentence words*
        ],
        [
            *len(batch_data) lists of sentence word features*
        ],
        [
            *len(batch_data) lists of sentence word labels*
        ]
    ]
    """
    batch = [[], [], []]
    for item in batch_data:
        for ix, tt in enumerate(item):
            batch[ix].append(tt)
    return batch


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
        # note that all data is returned in plain text,
        # so encoding needs to bedone while training


if __name__ == "__main__":
    ds = FaceLandmarksDataset()
    dl = DataLoader(ds, batch_size=2, collate_fn=collate,
                    shuffle=True, num_workers=4)
    for d in dl:
        [print(i) for i in d]
        break
    # print(ds[1])
