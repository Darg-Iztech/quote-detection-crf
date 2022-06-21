# Quote Detection CRF
Conditional Random Field (CRF) model for sequence labeling as the baseline model for Quote Detection task.

### Run CRF baseline:

**Example 1:**
```bash
python qtask_baseline.py --dataset='T50' --iter=500 --mode='xqx' --from_scratch
```

**Example 2:**
```bash
python qtask_baseline.py --dataset='MOVIE' --iter=500 --mode='pqn'
```

### Calculate ROUGE scores manually:

```bash
pip install rouge

rouge -f data/pred_quotes.txt data/true_quotes.txt --avg
```

### Results for 500 iterations:

| Metric | T50-PQN | T50-XQX | MOV-PQN | MOV-XQX |
| ------ | ------- | ------- | ------- | ------- |
| F1     | 0.99833 | 0.00000 | 0.99940 | 0.00119 |
| R1     | 0.65022 | 0.00000 | 0.48328 | 0.00105 |
| R2     | 0.60791 | 0.00000 | 0.44584 | 0.00104 |
| RL     | 0.64034 | 0.00000 | 0.48186 | 0.00105 |

### References:

https://github.com/arielsho/SemEval-2020-Task-5

https://github.com/pltrdy/rouge
