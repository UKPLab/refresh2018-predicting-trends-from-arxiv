import argparse
import numpy as np

from scipy.stats import pearsonr

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import load_model
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser(description='train models')
parser.add_argument('model', help="mlp or reg", type=str)
parser.add_argument('representation', help="tfidf or embeddings", type=str)
parser.add_argument('trainfile', type=str)
parser.add_argument('trainlabelsfile', type=str)
parser.add_argument('-tf', '--testfile', type=str, default=None)
parser.add_argument('-tl', '--testlabelsfile', type=str, default=None)
parser.add_argument('-n', '--nummodels', type=int, default=100)
parser.add_argument('-log', '--log_labels', action="store_true")
parser.add_argument('--textpart', help="abstract or title",
                    type=str, default='abstract')
parser.add_argument('-dev', '--devsplit', type=float, default=0.0)


def main():
    """
    python3 train_models.py -h
    Examples:
        python3 train_models.py mlp embeddings data/train/abstractunisent.train data/train/unisentlabel.train.real -tf data/train/abstractunisent.test -tl data/train/unisentlabel.test.real  -log -dev 0.1
        python3 train_models.py mlp embeddings data/train/abstractinfersent.train data/train/infersentlabel.train.real -tf data/train/abstractinfersent.test -tl data/train/infersentlabel.test.real  -log -dev 0.1
        python3 train_models.py reg tfidf data/train/abstract.train data/train/label.train.real -tf data/train/abstract.test -tl data/train/label.test.real -log -dev 0.1
        python3 train_models.py mlp embeddings data/train/titleunisent.train data/train/unisentlabel.train.real -tf data/train/titleunisent.test -tl data/train/unisentlabel.test.real  -log -dev 0.1
        python3 train_models.py mlp embeddings data/train/titleinfersent.train data/train/infersentlabel.train.real -tf data/train/titleinfersent.test -tl data/train/infersentlabel.test.real  -log -dev 0.1
        python3 train_models.py reg tfidf data/train/title.train data/train/label.train.real -tf data/train/title.test -tl data/train/label.test.real -log -dev 0.1
    """
    args = parser.parse_args()

    trainfile = args.trainfile
    labelfile = args.trainlabelsfile
    testfile = args.testfile
    testlabelfile = args.testlabelsfile

    save_vocab()
    nummodels = args.nummodels
    regression = True
    devsplit = args.devsplit
    alphalow, alphahigh = 0.1, 1.0

    train = []
    labels = []
    with open(trainfile, 'r') as f:
        if args.representation == 'tfidf':
            for line in f:
                train.append(line.strip())

            X = vectorize(train)
        else:
            for line in f:
                train.append(line.strip().split())
            X = np.array(train,dtype='float64')
    with open(labelfile) as f:
        for line in f:
            labels.append(int(line.strip()))

    y = np.array(labels,dtype='int32')
    if args.log_labels:
        y = np.ma.log(y).filled(0)
    if testfile:
        test = []
        labelstest = []

        with open(testfile, 'r') as f:
            if args.representation == 'tfidf':
                for line in f:
                    test.append(line.strip())

                X_test = vectorize(test)
            else:
                for line in f:
                    test.append(line.strip().split())
                X_test = np.array(test,dtype='float64')

        with open(testlabelfile) as f:
            for line in f:
                labelstest.append(int(line.strip()))
        y_test = np.array(labelstest,dtype='int32')
        if args.log_labels:
            y_test = np.ma.log(y_test).filled(0)



    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=devsplit)

    if args.model == 'reg':
        for i in range(nummodels):
            filename = 'models/regression/reg%i.%s.model' % (i, args.textpart)
            a = np.random.uniform(low=alphalow, high=alphahigh, size=None)
            model = regression_model(X_train, y_train, alpha=a)
            joblib.dump(model, filename)
            if devsplit > 0:
                pred = model.predict(X_dev)
                print(a, evaluate(y_dev, pred))

            if testfile:
                pred = model.predict(X_test)
                print('test:', evaluate(y_test, pred))

    if args.model == 'mlp':
        for i in range(nummodels):
            filename = 'models/regression/mlp%i.%s.model' % (i, args.textpart)
            a = np.random.uniform(low=alphalow, high=alphahigh, size=None)
            hidden_layer_size = np.random.choice([100])
            dropout = np.random.choice(np.arange(0.1, 0.31, 0.05))
            activation = np.random.choice(['relu'])

            model = mlp(np.array(X_train), np.array(y_train), np.array(X_dev), np.array(y_dev), filename,
                        regression=regression,
                        hidden_layers=1, hidden_layer_size=hidden_layer_size, dropout=dropout, activation=activation)

            if devsplit > 0:
                pred = [x[0] for x in model.predict(np.array(X_dev)).tolist()]
                print(a, evaluate(y_dev, pred))

            if testfile:
                pred = [x[0] for x in model.predict(np.array(X_test)).tolist()]
                print('test:', evaluate(y_test, pred))


def save_vocab():
    """
        Saves the vocab of the trainings files to reuse for tfidf
    """
    files = ['data/train/abstract.train', 'data/train/title.train']
    words = set()
    for file in files:
        with open(file) as f:
            for line in f:
                for word in line.strip().split():
                    words.add(word)

    with open('data/vocab.dat', 'w') as f:
        for word in words:
            f.write(word)
            f.write('\n')


def load_vocab():
    """
    loads the vocab
    Returns: list of words

    """
    words = []
    with open('data/vocab.dat') as f:
        for line in f:
            words.append(line.strip())
    return words


def vectorize(documents):
    """
    encodes documents with tfidf
    Args:
        documents: list of documents as strings

    Returns: Sparse matrix

    """
    vocab = load_vocab()
    vec = TfidfVectorizer(
        lowercase=False, ngram_range=(1, 1), vocabulary=vocab)

    return vec.fit_transform(documents)



def regression_model(X, y, alpha=.5):
    """
        trains a simple ridge regession model
    Args:
        X:
        y:
        alpha:

    Returns: model

    """
    reg = linear_model.Ridge(alpha=alpha, fit_intercept=True)
    # reg = linear_model.Lasso(alpha = alpha,fit_intercept = True)
    reg.fit(X, y)
    return reg


def mlp(X, y, X_dev, y_dev, filename, regression=True, hidden_layers=1, hidden_layer_size=100, dropout=0.25,
        activation='relu', batch_size=10, maxepochs=100):
    input_shape = X.shape
    inputs = Input(shape=(input_shape[1],))
    x = Dense(hidden_layer_size, activation=activation)(inputs)
    x = Dropout(dropout)(x)
    for i in range(hidden_layers - 1):
        x = Dense(hidden_layer_size, activation=activation)(x)
        x = Dropout(dropout)(x)

    if regression:
        checkpointer = ModelCheckpoint(
            filename, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)

        out = Dense(1, activation='relu')(x)
        model = Model(inputs=inputs, outputs=out)
        model.compile(optimizer='adam', loss='mean_squared_error')
    else:
        checkpointer = ModelCheckpoint(
            filename, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False)

        classes = set()
        for label in (y.tolist() + y_dev.tolist()):
            classes.add(label)
        num_classes = len(classes)
        out = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=out)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

    model.fit(X, y, validation_data=(X_dev, y_dev), callbacks=[checkpointer, es], batch_size=batch_size,
              epochs=maxepochs, verbose=0)
    model = load_model(filename)
    return model




def evaluate(true, pred, verbose=False):
    # acc = accuracy_score(true,pred)
    if verbose:
        for t, p in zip(true, pred):
            print(t, p)
    corr, p = pearsonr(true, pred)
    return corr, p


if __name__ == '__main__':
    main()
