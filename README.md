# Predicting Research Trends From Arxiv

This repository contains selected code and data for our [ReFresh](http://refresh.kmi.open.ac.uk/) workshop paper on [Predicting Research Trends From Arxiv](https://arxiv.org/abs/1903.02831).

## Citation

```
@inproceedings{Eger_arixv:2018,
           month = {December},
           title = {Predicting Research Trends From Arxiv},
            year = {2018},
       booktitle = {1st Workshop on Reframing Research},
          author = {Steffen Eger and Chao Li and Florian Netzer and Iryna Gurevych},
             url = {https://arxiv.org/abs/1903.02831},
             location = {Bonn, Germany}
}
```
> **Abstract:** We perform trend detection on two datasets of Arxiv papers, derived from its machine learning (cs.LG) and natural language processing (cs.CL) categories. Our approach is bottom-up: we first rank papers by their normalized citation counts, then group top-ranked papers into different categories based on the tasks that they pursue and the methods they use. We then analyze these resulting topics. We find that the dominating paradigm in cs.CL revolves around \emph{natural language generation} problems and those in cs.LG revolve around \emph{reinforcement learning} and \emph{adversarial principles}. By extrapolation, we predict that these topics will remain lead problems/approaches in their fields in the short- and mid-term.  

Contact person: Steffen Eger, eger@ukp.informatik.tu-darmstadt.de, Florian Netzer

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions. 

## Project Description

This project creates a list of the most influential papers
(defined by having a high [z-score](https://arxiv.org/abs/1310.8220))
in an [Arxiv comuter science category](https://arxiv.org/corr/home).

### Download the database and update citations and z-scores:
To initialize the database/download new papers to it, run:
```bash
python3 update_dataset.py -np 2017-05-01
```
This also downloads  citation information from semantic scholar and calculates the z-scores for the new papers.
To update citations and z-scores for the papers already in the dataset run:
```bash
python3 update_dataset.py -um 2017-05-01
```

Use the --category argument to change the [category](https://arxiv.org/corr/home) (default cs.LG).
In this case the papers will start from 2017-05-01(YYYY-MM-DD). To keep the database update you can just run `update_database.py`
with the `--newpapersstartdate` (-np) set to the date you last ran it and then update with `--updatemetastartdate` (-um) the papers for the time you are interested in.

### Get the top papers:
To print the papers with the top z-scores see:
```
python3 top_papers.py -h
```
For example, to print the top 100 papers starting from the first may 2017 and save it to a txt file run:
```
python3 top_papers.py 2017-05-01 100 > top100papers.txt
```

## Human annotations
Our human classification of the top-Arxiv papers into categories _method_, _task_, and _goals_ can be found in the directory `human_annotations`

## Citation Count Prediction as Regression on Text Embeddings
**(This is not part of the paper, but nonetheless included in this repository.)**

Step 1: get the text embeddings (for Infersent first downlod the [repository](https://github.com/facebookresearch/InferSent) and copy the models.py to this project):
```
python3 encode_sentences.py -h
```
Step 2: Trainigs data:
```
python3 get_train.py -h
```
Step 3: Train models:
```
python3 train_models.py -h
```
Step 4: Make predictions:
```
python3 make_predictions.py -h
```
Print predictions:
```
python3 top_papers.py 2017-05-01 100 -pred
```
For an example for step 2-4 see train_models.sh
