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

### Results for 500 iterations:

| Metric | T50-PQN  | MOV-PQN |
| ------ | -------  | ------- |
| F1     | 0.97108  | 1.00000 |
| R1     | 0.25349  | 0.25388 |
| R2     | 0.15925  | 0.20371 |
| RL     | 0.23071  | 0.25035 |

### References:

https://github.com/arielsho/SemEval-2020-Task-5

https://github.com/pltrdy/rouge
