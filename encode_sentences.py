"""Encodes textparts as InferSent embeddings

Attributes:
    glovePath (str): path to glove embeddings
    infersentpath (str): path to infersent.allnli.pickle
"""
import datahandler
from mydate import valid_date

import argparse

import torch
import os.path
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

infersentpath = '/home/eger/projects/InferSent/encoder/infersent.allnli.pickle'
glovePath = "/home/eger/projects/InferSent/glove.840B.300d.txt.startsym"

infersentpath = 'InferSent/encoder/infersent.allnli.pickle'
glovePath = 'InferSent/dataset/GloVe/glove.840B.300d.txt'

parser = argparse.ArgumentParser(description='encode sentences in the db')
parser.add_argument('--embedding',help= "infersent or unisent or both", type=str, default="both")
parser.add_argument('--startdate', help="format YYYY-MM-DD or None",
                    type=valid_date, default=None)
parser.add_argument('--textpart', help="abstract or title or both",
                    type=str, default="both")
parser.add_argument('--usegpu', type=bool, default=False)



def main():
    """Usage: python3 encode_sentences.py -h
    """
    args = parser.parse_args()
    if args.embedding == "infersent" or  args.embedding == "both":
        if args.textpart == "both":
            infersent_encode_papers(startdate=args.startdate, textpart="title", usegpu=args.usegpu)
            infersent_encode_papers(startdate=args.startdate, textpart="abstract", usegpu=args.usegpu)
        else:
            infersent_encode_papers(startdate=args.startdate, textpart=args.textpart, usegpu=args.usegpu)
    if args.embedding == "unisent" or args.embedding == "both":
        if args.textpart == "both":
            unisent_encode(startdate=args.startdate, textpart="title")
            unisent_encode(startdate=args.startdate,textpart="abstract")
        else:
            unisent_encode(startdate=args.startdate, textpart=args.textpart)


def infersent_encode_papers(startdate=None, textpart='abstract', usegpu=False, outdir='data/'):
    """Encodes all papers as infersent embeddings for one text part. Creates a csv file with arxivids and sentence vectors
    
    Args:
        startdate (datetime.datetime, optional): Startdate. If None all papers are encoded
        textpart (str, optional): abstract or title
        usegpu (bool, optional): set to True if run with cuda
        outdir (str, optional): dir to write the embeddings to
    """
    outfile = outdir + 'infersent_%s.csv' % textpart


    sentences = []
    vocab_texts = []
    index_arxivid = {}
    for i, paper in enumerate(datahandler.get_all_papers_iterator()):
        if startdate:
            vocab_texts.append(paper[textpart].lower())
            if paper['created'] >= startdate:
                index_arxivid[i] = paper['arxivid']
                sentences.append(paper[textpart].lower())
        else:
            index_arxivid[i] = paper['arxivid']
            sentences.append(paper[textpart].lower())


    if startdate:
        embeddings = infersent_encode_texts(vocab_texts,sentences,usegpu=False)
    else:
        embeddings = infersent_encode_texts(sentences, sentences, usegpu=False)

    wa = 'a' if os.path.exists(outfile) else 'w'
    with open(outfile, wa) as f:
        for i, arxivid in index_arxivid.items():
            f.write(" ".join([arxivid] + list(map(str, embeddings[i, :]))))
            f.write("\n")

def infersent_encode_texts(vocab_texts, texts, usegpu=False):
    if usegpu:
        infersent = torch.load(infersentpath)
    else:
        infersent = torch.load(
            infersentpath, map_location=lambda storage, loc: storage)

    infersent.set_glove_path(glovePath)

    infersent.build_vocab(vocab_texts, tokenize=True)

    return infersent.encode(texts, tokenize=True)


def unisent_encode(startdate=None, textpart='abstract', outdir='data/'):
    """Encodes all papers as unisent embeddings for one text part. Creates a csv file with arxivids and sentence vectors

    Args:
        startdate (datetime.datetime, optional): Startdate. If None all papers are encoded
        textpart (str, optional): abstract or title
        outdir (str, optional): dir to write the embeddings to
    """
    outfile = outdir + 'unisent_%s.csv' % textpart

    embed = hub.Module("https://tfhub.dev/google/"
                       "universal-sentence-encoder/1")
    texts = []
    index_arxivid = {}

    for i, paper in enumerate(datahandler.get_all_papers_iterator()):
        if startdate:
            if paper['created'] >= startdate:
                index_arxivid[i] = paper['arxivid']
                texts.append(paper[textpart].lower())
        else:
            index_arxivid[i] = paper['arxivid']
            texts.append(paper[textpart].lower())

    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        text_embeddings = session.run(embed(texts))

    wa = 'a' if os.path.exists(outfile) else 'w'
    with open(outfile, wa) as f:
        for i, embedding in enumerate(np.array(text_embeddings).tolist()):
            arxivid = index_arxivid[i]
            f.write(" ".join([arxivid] + list(map(str, embedding))))
            f.write("\n")


if __name__ == "__main__":
    main()
