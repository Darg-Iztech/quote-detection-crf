# Quote Detection CRF
Conditional Random Field (CRF) model for sequence labeling as the baseline model for Quote Detection task.

### Run CRF baseline:

**Example 1:**
```bash
python qtask_baseline.py --dataset='T50' --iter=500 --from_scratch
```

**Example 2:**
```bash
python qtask_baseline.py --dataset='MOVIE' --iter=500
```

### Calculate ROUGE scores manually:

```bash
pip install rouge

rouge -f data/pred_quotes.txt data/true_quotes.txt --avg
```

### Results for 500 iterations and 5-fold cross-validation:

| Metric | T50           | MOV          |
| ------ | ------------  | ------------ |
| R1     | 27.24 ± 0.19  | 26.42 ± 0.13 |
| R2     | 18.58 ± 0.26  | 20.72 ± 0.32 |
| RL     | 25.25 ± 0.22  | 25.75 ± 0.20 |


### References:

https://github.com/arielsho/SemEval-2020-Task-5

https://github.com/pltrdy/rouge
