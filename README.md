# CoRec
Custom implementation of "Context-Aware Retrieval-based Deep Commit Message Generation" paper for automatic commit messages generation.

The original code has been renewd and upgraded to the last libraries version then expanded with additional architectures choices including a Transformer encoder-decoder model. All details about different implementations evaluated and tests conducted in this codebase can be found in the followind documents:

[Documentation with approach explained](docs/main.pdf)

[Presentation of the project](docs/CoRec_project_discussion.pdf)

A trained model of our Transformer's implementation can be found [here](https://drive.google.com/file/d/1Jc21sLDebH17SsfATDaG2l6203_JGpYa/view?usp=sharing)


### Environment Dependencies

`python` >= 3.5

`pytorch` : 1.11

`torchtext` >= 0.11

`ConfigArgParse` : 1.5.3

`nltk` >= 3.4.5

`numpy` : 1.21.5

### Datasets
 (From CoRec search paper)Wang et al. dataset, crawled from 10000 repositories in Github: `data/top10000/merged/`
 
 The dataset from Jiang et al., 2017 and cleaned by Liu et al., 2018: `data/top1000/`
 
### Preprocess
`run_top1000.py` is the script for top-1000 dataset.
`run_top10000.py` is the script for top-10000 dataset.

For example:
```bash
$ python3 run_top1000.py preprocess
```

### Train
```bash
$ python3 run_top1000.py train
```

### Test
```bash
$ python3 run_top1000.py translate
```

The generated commit message will be saved in file: `data/output/1000test.out` or `data/output/10000test.out`
and the evaluation metrics will be returned in console.

### Evaluation
The script for exaluation: `evaluation/evaluate_res.py`

