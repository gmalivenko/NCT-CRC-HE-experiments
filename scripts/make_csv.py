from glob import glob
import os
import pandas as pd


PATH_TO_DATASET = "../data/NCT-CRC-HE-100K/"
PATH_TO_TEST_DATASET = "../data/CRC-VAL-HE-7K/"


def create_csv(input_dir, output_csv):
    samples = [i for i in glob(os.path.join(input_dir, '**/*')) if os.path.isfile(i)]
    label_map = {
        'ADI': 0,
        'BACK': 1,
        'DEB': 2,
        'LYM': 3,
        'MUC': 4,
        'MUS': 5,
        'NORM': 6,
        'STR': 7,
        'TUM': 8,
    }

    y = [label_map[os.path.basename(i).split('-')[0]] for i in samples]

    columns = ['img', 'label']
    df = pd.DataFrame(zip(samples, y), columns=columns)
    df.to_csv(output_csv)


if __name__ == '__main__':
    create_csv(PATH_TO_DATASET, 'train.csv')
    create_csv(PATH_TO_TEST_DATASET, 'test.csv')