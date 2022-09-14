"""
    Process the raw input documents and create the parabel data files.
"""
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer

from convert_format import split_data_into_fts_lbls


def write_data(filename, features, labels, header=True):
    """Write data in sparse format

    Arguments
    ---------
    filename: str
        output file name
    features: csr_matrix
        features matrix
    labels: csr_matix
        labels matrix
    """
    if header:
        with open(filename, 'w') as f:
            out = "{} {} {}".format(
                features.shape[0], features.shape[1], labels.shape[1])
            print(out, file=f)
        with open(filename, 'ab') as f:
            dump_svmlight_file(features, labels, f, multilabel=True)
    else:
        with open(filename, 'wb') as f:
            dump_svmlight_file(features, labels, f, multilabel=True)


def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    token_pattern = r"(?u)\b\w\w+\b"
    # Filter df['title'] by token_pattern. This is because, after preprocessing this is giving empty strings for some
    # datapoints, thus causing TfIdfVectorizer to return empty feature vectors.
    df = df[df['title'].str.contains(token_pattern)]
    # df = df[:10000]
    labels = df["labels"].values
    labels = [[sub_label for sub_label in label_string.split()] for label_string in labels]
    multilabel_binarizer = MultiLabelBinarizer(sparse_output=True)
    multilabel_binarizer.fit(labels)

    def to_indicator_matrix(some_df):
        some_df_labels = some_df["labels"].values
        some_df_labels = [[l for l in label_string.split()] for label_string in some_df_labels]
        return multilabel_binarizer.transform(some_df_labels)

    tf_idf = TfidfVectorizer(max_features=25000, token_pattern=token_pattern)
    tf_idf.fit(df["title"].values)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    X_train = train_df["title"].values
    X_train = tf_idf.transform(X_train)
    y_train = to_indicator_matrix(train_df)

    X_test = test_df["title"].values
    X_test = tf_idf.transform(X_test)
    y_test = to_indicator_matrix(test_df)

    return X_train, y_train, X_test, y_test


def clean_exported_file(f_name, ):
    # Clean the exported data files whenever there are no features.
    with open(f_name, 'r') as f:
        lines = f.readlines()

    with open(f_name, 'w') as f:
        no_of_features, no_of_unique_labels = lines[0].split()[1], lines[0].split()[2]
        good_lines = [line for line in lines[1:] if len(line.split()) >= 2]
        # Update counts no_of_samples, no_of_features, no_of_labels in the first line of the file.
        good_lines.insert(0, '{} {} {}\n'.format(len(good_lines), no_of_features, no_of_unique_labels))
        del lines
        for line in good_lines:
            f.write(line)


def create_parabel_data_files(dataset, raw_data_file):
    """
    Creates the parabel data files from the raw data file.
    :param dataset:
    :param raw_data_file:
    :return:
    """
    # data_dir = dataset + '/Data'
    # trn_ft_file = data_dir + '/trn_X_Xf.txt'
    # trn_lbl_file = data_dir + '/trn_X_Y.txt'
    # tst_ft_file = data_dir + '/tst_X_Xf.txt'
    # tst_lbl_file = data_dir + '/tst_X_Y.txt'

    # Change the path according to where the source data file is.

    trn_ofname = dataset + '/{}_trn.txt'.format(dataset)
    tst_ofname = dataset + '/{}_tst.txt'.format(dataset)
    # Read data and create features
    trn_features, trn_labels, tst_features, tst_labels = load_dataset(raw_data_file)

    # write the data
    print('Writing data into files - {} and {}...'.format(trn_ofname, tst_ofname))
    write_data(trn_ofname, trn_features, trn_labels)
    write_data(tst_ofname, tst_features, tst_labels)
    print('Writing data into files... Done')

    print('Cleaning exported data...')
    clean_exported_file(trn_ofname)
    clean_exported_file(tst_ofname)
    print('Cleaning exported data... done')

    return trn_ofname, tst_ofname


def create_parabel_data_files_v1(dataset, raw_data_file):
    """
    Creates the parabel data files from the raw data file and splits them into separate features / label files.
    :param dataset:
    :param raw_data_file:
    :return:
    """
    data_dir = dataset + '/Data'
    trn_ft_file = data_dir + '/trn_X_Xf.txt'
    trn_lbl_file = data_dir + '/trn_X_Y.txt'
    tst_ft_file = data_dir + '/tst_X_Xf.txt'
    tst_lbl_file = data_dir + '/tst_X_Y.txt'

    # Change the path according to where the source data file is.

    trn_ofname = dataset + '/{}_trn.txt'.format(dataset)
    tst_ofname = dataset + '/{}_tst.txt'.format(dataset)
    # Read data and create features
    trn_features, trn_labels, tst_features, tst_labels = load_dataset(raw_data_file)

    # write the data
    print('Writing data into files - {} and {}...'.format(trn_ofname, tst_ofname))
    write_data(trn_ofname, trn_features, trn_labels)
    write_data(tst_ofname, tst_features, tst_labels)
    print('Writing data into files... Done')

    print('Cleaning exported data...')
    clean_exported_file(trn_ofname)
    clean_exported_file(tst_ofname)
    print('Cleaning exported data... done')

    print('Split data into features and labels...')
    split_data_into_fts_lbls(data_fname=trn_ofname, data_fts_fname=trn_ft_file,
                             data_lbls_fname=trn_lbl_file)

    split_data_into_fts_lbls(data_fname=tst_ofname, data_fts_fname=tst_ft_file,
                             data_lbls_fname=tst_lbl_file)
    print('Split data into features and labels... Done')
