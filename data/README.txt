
Training set: First 80% of the dataset
Test set: Last 20% of the dataset
----------------------------------------------------------------------------------
matching sentence:
     the sentence that is most similar to the given quote in the 10-sentence-span
----------------------------------------------------------------------------------
task2prev_train.csv & task2prev_test.csv:
     antecedent = previous sententes of matching sentence
     consequent = matching sentence
----------------------------------------------------------------------------------
task2next_train.csv & task2next_test.csv:
     antecedent = matching sentence
     consequent = next sententes of matching sentence
----------------------------------------------------------------------------------
task2both_train.csv & task2both_test.csv:
     antecedent = previous + next sententes of matching sentence
     consequent = matching sentence
----------------------------------------------------------------------------------