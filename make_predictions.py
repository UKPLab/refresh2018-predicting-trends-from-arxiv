import argparse

from nltk.tokenize import word_tokenize
from pathlib import Path
from mydate import valid_date
from encode_sentences import infersent_encode_texts
from sklearn.externals import joblib
import datahandler
from train_models import vectorize
from keras.models import load_model
import numpy as np

parser = argparse.ArgumentParser(description='train models')
parser.add_argument("startdate", help="startdate- format YYYY-MM-DD",
                    type=valid_date, default=None)
parser.add_argument('--textpart', help="abstract or title",
                    type=str, default='abstract')


def main():
    args = parser.parse_args()
    papers = datahandler.get_papers_after(args.startdate)

    texts = [paper[args.textpart] for paper in papers]
    predictions = predict_texts(texts, textpart=args.textpart, regression=True)

    for pred,paper in zip(predictions,papers):
        print(pred)
        paper["prediction_%s" % args.textpart] = pred
        datahandler.update_paper(paper["arxivid"], paper)


def predict_texts(texts, textpart='abstract', regression=True):
    """
        TODO test for classification
        makes a prediction using pre trained models. Takes a majority-vote/mean from all models
    Args:
        texts(list str): text
        textpart(str): abstract or title
        regression(bool): True to use regression models false to use classification models

    Returns:
        float: the prediction
    """
    vocab_texts = []
    for i, paper in enumerate(datahandler.get_all_papers_iterator()):
        vocab_texts.append(paper[textpart].lower())
    embeddings = infersent_encode_texts(vocab_texts, texts)

    texts = [str(word_tokenize(text.lower())) for text in texts]
    x_tfidf = vectorize(texts)

    if regression:
        pathlist = Path('models/regression/').glob('*.%s.model' % textpart)
    else:
        pathlist = Path('models/classification/').glob('*.%s.model' % textpart)
    predictions = []
    for path in pathlist:
        if 'mlp' in str(path):
            model = load_model(str(path))
            p = model.predict(embeddings).flatten()
        else:
            model = joblib.load(str(path))
            p = model.predict(x_tfidf).flatten()
        print(p)
        predictions.append(p)
    predictions = np.array(predictions)

    if regression:
        return np.mean(predictions, axis=1).tolist()
    else:
        pred = []
        predictions = predictions.transpose()
        for p in predictions:
            (values, counts) = np.unique(p, return_counts=True)
            ind = np.argmax(counts)
            pred.append(values[ind])
        return  pred


if __name__ == '__main__':
    main()
