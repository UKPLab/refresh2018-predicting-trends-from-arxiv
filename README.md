This project finds creates a list of the most influential papers
(defined by having a high [z-score](https://arxiv.org/abs/1310.8220))
in a [arxiv comuter science category](https://arxiv.org/corr/home).

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

Use the --category argument to change the [category](https://arxiv.org/corr/home)(default cs.LG).
In this case the papers will start from 2017-05-01(YYYY-MM-DD). To keep the database update you can just run update_database.py
with the --newpapersstartdate (-np) set to the date you last ran it and then update with --updatemetastartdate (-um) the papers for the time you are interested in.
### Get the top papers:
To print the papers with the top z-scores see:
```
python3 top_papers.py -h
```
For example, to print the top 100 papers starting from the first may 2017 and save it to a txt file run:
```
python3 top_papers.py 2017-05-01 100 > top100papers.txt
```
If you just want to have the top papers list you can also use [this](https://git.ukp.informatik.tu-darmstadt.de/netzer/top-arxiv-papers).

### Citation Count Prediction as Regression on Text Embeddings
Step 1: get the text embeddings(for Infersent first downlod the [repository](https://github.com/facebookresearch/InferSent) and copy the models.py to this project):
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