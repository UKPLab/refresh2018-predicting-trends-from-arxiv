import datahandler
from mydate import valid_date

import argparse
from nltk.tokenize import word_tokenize

parser = argparse.ArgumentParser(description='Write the files for training and testing models')
parser.add_argument('startdate', help="format YYYY-MM-DD", type=valid_date)
parser.add_argument('enddate', help="format YYYY-MM-DD", type=valid_date)
parser.add_argument('-p', '--textpart', help="abstract or title or all (default all)", type=str, default='all')
parser.add_argument('--outdir', help="directory to write the files to", type=str, default='data/train/')
parser.add_argument('--binning_steps', nargs='+', type=int, default=[1, 3])
parser.add_argument('-ts', '--teststartdate', help="format YYYY-MM-DD, if not set no test set will be written",
                    type=valid_date, default=None)
parser.add_argument('-te', '--testenddate', help="format YYYY-MM-DD, if not set no test set will be written",
                    type=valid_date, default=None)


def main():
    """Usage: python3 get_train.py -h
    """
    args = parser.parse_args()

    if args.textpart == 'all':

        write_train_data(args.startdate, args.enddate, textpart='abstract', binning_steps=args.binning_steps,
                         teststartdate=args.teststartdate, testenddate=args.testenddate)
        write_train_data(args.startdate, args.enddate, textpart='title', binning_steps=args.binning_steps,
                         teststartdate=args.teststartdate, testenddate=args.testenddate)
    else:
        write_train_data(args.startdate, args.enddate, textpart=args.textpart, binning_steps=args.binning_steps,
                         teststartdate=args.teststartdate, testenddate=args.testenddate)


def bin(label, steps=[1, 3]):
    """Put a label into bins between steps
    
    Args:
        label (int): input label
        steps (list, optional): where to seperate the bins
    
    Returns:
        int: binned label
    """
    i = 0
    for step in steps:
        if label < step:
            return i
        else:
            i += 1
    return i


def write_train_data(startdate, enddate, textpart='abstract', out_dir='data/train/', binning_steps=[1, 3],
                     teststartdate=None, testenddate=None):
    """writes train data for one textpart for a timewindow to a file
    
    Args:
        startdate (datetime.datetime): startdate
        enddate (datetime.datetime): enddate
        textpart (str): abstract or title
        out_dir (str): path to a directory to write the files in
        binning_steps (list, optional): steps for the binning
    """

    write_data(startdate, enddate, textpart, out_dir, binning_steps, 'train')
    write_embeddingdata(startdate, enddate, textpart, out_dir, binning_steps, 'train', 'infersent')
    write_embeddingdata(startdate, enddate, textpart, out_dir, binning_steps, 'train', 'unisent')

    if teststartdate and testenddate:
        write_data(teststartdate, testenddate, textpart, out_dir, binning_steps, 'test')
        write_embeddingdata(teststartdate, testenddate, textpart, out_dir, binning_steps, 'test', 'infersent')
        write_embeddingdata(teststartdate, testenddate, textpart, out_dir, binning_steps, 'test', 'unisent')


def write_data(startdate, enddate, textpart, out_dir, binning_steps, trainortest):
    papers = datahandler.get_papers_in_timewindow(startdate, enddate)

    x = []
    y_real = []

    for paper in papers:
        citations = [c for c in paper['citations'] if int(c['year']) <= (paper['created'].year + 1)]
        text = paper[textpart]
        text = word_tokenize(text.lower())
        x.append(text)
        label = len(citations)
        y_real.append(label)

    p = out_dir + textpart
    xfile = p + '.' + trainortest
    labelfilereal = out_dir + 'label.%s.real' % trainortest
    labelfilebinned = out_dir + 'label.%s.binned' % trainortest

    with open(xfile, 'w') as f:
        for text in x:
            f.write(' '.join(text) + '\n')

    with open(labelfilereal, 'w') as fr, open(labelfilebinned, 'w') as fb:
        for label in y_real:
            fr.write(str(label))
            fr.write('\n')
            fb.write(str(bin(label, steps=binning_steps)))
            fb.write('\n')


def write_embeddingdata(startdate, enddate, textpart, out_dir, binning_steps, trainortest, embedding):
    papers = datahandler.get_papers_in_timewindow(startdate, enddate)
    arxivid_embedding = datahandler.get_arxivid_embedding(embedding=embedding, textpart=textpart)

    x = []
    y = []
    for paper in papers:
        citations = [c for c in paper['citations'] if int(c['year']) <= (paper['created'].year + 1)]
        label = len(citations)
        if paper['arxivid'] in arxivid_embedding:
            x.append(arxivid_embedding[paper['arxivid']])
            y.append(label)

    p = out_dir + textpart
    xfile = p + '%s.' % embedding + trainortest
    labelfilereal = out_dir + '%slabel.%s.real' % (embedding, trainortest)
    labelfilebinned = out_dir + '%slabel.%s.binned' % (embedding, trainortest)

    with open(xfile, 'w') as f:
        for embedding in x:
            f.write(' '.join(str(x) for x in embedding.tolist()))
            f.write('\n')

    with open(labelfilereal, 'w') as fr, open(labelfilebinned, 'w') as fb:
        for label in y:
            fr.write(str(label))
            fr.write('\n')
            fb.write(str(bin(label, steps=binning_steps)))
            fb.write('\n')


if __name__ == "__main__":
    main()
