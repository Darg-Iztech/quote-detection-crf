'''
Task2 baseline model
'''
import csv
from re import T
import nltk
import pycrfsuite
import ast
import numpy as np
import sys
import argparse
from os.path import exists
from rouge import FilesRouge
from sklearn.model_selection import KFold
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def word2features(doc, i):

    '''
    For the input word, create its feature list
    :param doc: tokenized sentence
    :param i: index
    :return: features: the word's feature list
    '''
    word = doc[i][0]
    postag = doc[i][1]

    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]


    if i > 0:
        word1 = doc[i - 1][0]
        postag1 = doc[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1
        ])
    else:

        features.append('BOS')



    if i < len(doc) - 1:
        word1 = doc[i + 1][0]
        postag1 = doc[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1
        ])
    else:
        features.append('EOS')

    return features


def extract_features(doc):
    '''
    extract the features in one sentence
    :param doc: tokenized sentence
    :return: a feature list for every word in this sentence
    '''
    return [word2features(doc, i) for i in range(len(doc))]


def label_data(fpath):
    print('\n\nLabeling {}\n\n'.format(fpath))
    docs = []
    with open(fpath, 'r', encoding="utf-8", errors="ignore") as readFile:
        reader = csv.reader(readFile)  # csv reader

        lines = list(reader)

        for line in lines[1:]:
            sent_id=line[0]
            sent=line[1]
            prev=line[2]
            quot=line[3]
            next=line[4]

            sent_token=nltk.tokenize.word_tokenize(sent)
            prev_token=nltk.tokenize.word_tokenize(prev)
            quot_token=nltk.tokenize.word_tokenize(quot)
            next_token=nltk.tokenize.word_tokenize(next)

            prev_begin_index = 0
            prev_end_index = len(prev_token)
            quot_begin_index = prev_end_index + 1
            quot_end_index = quot_begin_index + len(quot_token)
            next_begin_index = quot_end_index + 1
            next_end_index = next_begin_index + len(next_token)

            label=[]

            for i in range(len(prev_token)):
                label.append('P')
            for i in range(len(quot_token)):
                label.append('Q')
            for i in range(len(next_token)):
                label.append('N')

            doc=[]

            for i in range(len(sent_token)):
                doc.append(tuple((sent_token[i], label[i])))

            doc.insert(0, sent_id)
            docs.append(doc)
            print('sent_id {} is OK!'.format(sent_id))

    readFile.close()

    return docs


def postag_data(labeled_data, fpath):

    print('\nPos-tagging dataset...\n')

    data = []
    for _, labeled in enumerate(labeled_data):
        tokens = [w for (w, label) in labeled[1:]]
        tagged = nltk.pos_tag(tokens)
        tagged_labeled_data = [(w, pos, label) for (w, label), (word, pos) in zip(labeled[1:], tagged)]
        tagged_labeled_data.insert(0, labeled[0])
        data.append(tagged_labeled_data)

    with open(fpath, 'wt', encoding='utf-8', newline='') as save_data:
        cw=csv.writer(save_data)
        for row in data:
            cw.writerow(row)

    return data


def get_labels(doc):
    '''
    extract the labels in one sentence
    :param doc: tokenized sentence
    :return: a label list for every word in this sentence
    '''
    return [label for (token, postag, label) in doc]


def get_coordinate(x_data, y_data):

    coordinates_sentence = []
    coordinates_all_sentences = []

    for index_sentence, y_sentence in enumerate(y_data):

        current_index = 0
        prev_begin = 0
        prev_end = 0
        quot_begin = 0
        quot_end = 0
        next_begin = 0
        next_end = 0

        for index_word, y_word in enumerate(y_sentence[1]):
            token = x_data[index_sentence][1][index_word][1].split('word.lower=')[1]
            offset = len(token) + 1  # we add 1 for whitespace coming after the token

            if y_word == 'P' or (y_word == 'X' and quot_begin == 0):
                current_index += offset
                continue
            elif y_word == 'Q':
                if quot_begin == 0:
                    prev_end = current_index
                    quot_begin = current_index
                current_index += offset
                continue
            elif y_word == 'N' or (y_word == 'X' and quot_begin != 0):
                if next_begin == 0:
                    quot_end = current_index
                    next_begin = current_index
                current_index += offset
                continue

        next_end = current_index
                
        coordinates_sentence = [y_sentence[0], prev_begin, prev_end, quot_begin, quot_end, next_begin, next_end]

        coordinates_all_sentences.append(coordinates_sentence)

    return coordinates_all_sentences


def format_judge(submission):
    '''
    judge if the submission file's format is legal
    :param submission: submission file
    :return: False for illegal
             True for legal
    '''
    # submission: [sentenceID,antecedent_startid,antecedent_endid,consequent_startid,consequent_endid]

    if submission[1] == '-1' or submission[2] == '-1':
        return False
    if (submission[3] == '-1' and submission[4] != '-1') or (submission[3] != '-1' and submission[4] == '-1'):
        return False
    if (int(submission[1]) >= int(submission[2])) or (int(submission[3]) > int(submission[4])):
        return False
    if not (int(submission[1]) >= -1 and int(submission[2]) >= -1 and int(submission[3]) >= -1 and int(submission[4]) >= -1):
        return False
    return True


def get_inter_id(submission_idx, truth_idx):

    sub_start = int(submission_idx[0])
    sub_end = int(submission_idx[1])
    truth_start = int(truth_idx[0])
    truth_end = int(truth_idx[1])
    if sub_end < truth_start or sub_start > truth_end:
        return False, -1, -1
    return True, max(sub_start, truth_start), min(sub_end, truth_end)


def metrics_task(submission_list, truth_list):
    # submission_list:  [[send_id, prev_start, prev_end, quot_start, quot_end, next_start, next_end], ...]
    # truth_list:       [[sent_id, sentence, prev_start, prev_end, quot_start, quot_end, next_start, next_end], ...]

    f1_score_all = []
    precision_all = []
    recall_all = []

    for i in range(len(submission_list)):
        assert submission_list[i][0] == truth_list[i][0]
        submission = submission_list[i]
        truth = truth_list[i]

        precision = 0
        recall = 0
        f1_score = 0

        if format_judge(submission):
            # truth processing
            sentence = truth[1]

            t_p_s = int(truth[2])       # truth_prev_startid
            t_p_e = int(truth[3])       # truth_prev_endid
            t_q_s = int(truth[4])       # truth_quot_startid
            t_q_e = int(truth[5])       # truth_quot_endid
            t_n_s = int(truth[6])       # truth_next_startid
            t_n_e = int(truth[7])       # truth_next_endid

            s_p_s = int(truth[2])       # submission_prev_startid
            s_p_e = int(truth[3])       # submission_prev_endid
            s_q_s = int(truth[4])       # submission_quot_startid
            s_q_e = int(truth[5])       # submission_quot_endid
            s_n_s = int(truth[6])       # submission_next_startid
            s_n_e = int(truth[7])       # submission_next_endid

            truth_prev_len = len(sentence[t_p_s : t_p_e].split())
            truth_quot_len = len(sentence[t_q_s : t_q_e].split())
            truth_next_len = len(sentence[t_n_s : t_n_e].split())
            truth_len = truth_prev_len + truth_quot_len + truth_next_len

            submission_prev_len = len(sentence[s_p_s : s_p_e].split())
            submission_quot_len = len(sentence[s_q_s : s_q_e].split())
            submission_next_len = len(sentence[s_n_s : s_n_e].split())
            submission_len = submission_prev_len + submission_quot_len + submission_next_len

            # intersection
            inter_prev_flag, inter_prev_startid, inter_prev_endid = get_inter_id([s_p_s, s_p_e], [t_p_s, t_p_e])
            inter_quot_flag, inter_quot_startid, inter_quot_endid = get_inter_id([s_q_s, s_q_e], [t_q_s, t_q_e])
            inter_next_flag, inter_next_startid, inter_next_endid = get_inter_id([s_n_s, s_n_e], [t_n_s, t_n_e])

            inter_prev_len = 0
            inter_quot_len = 0
            inter_next_len = 0
    
            if inter_prev_flag:
                inter_prev_len = len(sentence[inter_prev_startid : inter_prev_endid].split())
            if inter_quot_flag:
                inter_quot_len = len(sentence[inter_quot_startid : inter_quot_endid].split())
            if inter_next_flag:
                inter_next_len = len(sentence[inter_next_startid : inter_next_endid].split())
            
            inter_len = inter_prev_len + inter_quot_len + inter_next_len

            # calculate precision, recall, f1-score
            if inter_len > 0:
                precision = inter_len / submission_len
                recall = inter_len / truth_len
                f1_score = 2 * precision * recall / (precision + recall)

        precision_all.append(precision)
        recall_all.append(recall)
        f1_score_all.append(f1_score)

    f1_mean = np.mean(f1_score_all)
    precision_mean = np.mean(precision_all)
    recall_mean = np.mean(recall_all)
    return f1_mean, precision_mean, recall_mean,f1_score_all


def evaluate(truth_reader, submission_list, true_sentence):
    truth_list=[]
    not_em = 0
    for idx, line in enumerate(truth_reader):
        tmp = []
        submission_line = submission_list[idx]
        if line[0] != submission_line[0]:
            sys.exit("Sorry, the sentence id is not matched.")
        tmp.append(line[0])    # sentenceID
        tmp.append(true_sentence[idx][1])
        tmp.extend(line[1:])  # prev_start, prev_end, quot_start, quot_end, next_start, next_end
        truth_list.append(tmp)

    f1_mean, recall_mean, precision_mean,f1_score_all = metrics_task(submission_list, truth_list)

    return f1_mean, recall_mean, precision_mean, f1_score_all


def turn_list_to_str(sentence):
    '''
    recover list to string
    :param sentence: token list
    :return: string
    '''
    sentence_str=""
    for _, word in enumerate(sentence):
        if str(word).find('\'')==0 and len(str(word))!=1 and str(word[0]).find('\'')==-1:
            sentence_str+=('\''+" "+str(word[1:]))
        elif str(word).find('\'') == 0 and len(str(word)) != 1:
            sentence_str =sentence_str.rstrip ()+str(word)+" "
        elif str(word).find('\'')>0:
            sentence_str=sentence_str.rstrip()+str(word)+" "
        elif str(word)=='{' or str(word)=='}':
            sentence_str+=str(word)
        elif str(word)=='/' or str(word)==')'or str(word)=='\''or str(word)=='\"' or str(word)=='('  or str(word)=='’' or str(word)=='`' or str(word)=='‘':
            sentence_str=sentence_str.rstrip()+str(word)
        elif str(word)==',' or str(word)=='.' or str(word)=='!' or str(word)=='?' or str(word)==';' or str(word)==':':
            sentence_str=sentence_str.rstrip()+str(word)+" "
        else:
            sentence_str+=(str(word)+" ")
    return sentence_str.strip()


def turn_sentences_to_str(predict_sentences):
    '''
    recover token sequence to original sentence
    :param predict_sentences: token sequence
    :return: original sentence
    '''
    for sentence in predict_sentences:
        sentence[1]=turn_list_to_str(sentence[1])
        sentence[2]=turn_list_to_str(sentence[2])
        sentence[3]=turn_list_to_str(sentence[3])
        sentence[4]=turn_list_to_str(sentence[4])
    return predict_sentences


def get_predict_content(test_data_reader, y_test):

    predict_sentences=[]

    for index, predict_label in enumerate(y_test):
        predict_sentence = []
        predict_sentence.append(predict_label[0])
        prev=[]
        quot=[]
        next=[]
        sent=[]
        quot_is_passed=0
        for test_data in test_data_reader:
            if test_data[0]==predict_label[0]:
                for index,tag in enumerate(predict_label[1]):
                    if tag=='P' or (tag=='X' and quot_is_passed==0):
                        prev.append(test_data[index+1][0])
                        sent.append(test_data[index+1][0])
                        continue
                    if tag=='Q':
                        quot.append(test_data[index+1][0])
                        sent.append(test_data[index+1][0])
                        quot_is_passed=1
                        continue
                    if tag=='N' or (tag=='X' and quot_is_passed==1):
                        next.append(test_data[index +1][0])
                        sent.append(test_data[index+1][0])
                        continue

        predict_sentence.append(sent)
        predict_sentence.append(prev)
        predict_sentence.append(quot)
        predict_sentence.append(next)
        predict_sentences.append(predict_sentence)
    return predict_sentences


def read_postagged_data(fpath):
    pt_data=[]
    pt_line=[]
    with open(fpath,'r',encoding='utf-8',errors="ignore") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            pt_line=[ast.literal_eval(word)for word in row[1:]]
            pt_line.insert(0,row[0])
            pt_data.append(pt_line)
    return pt_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='qtask')
    parser.add_argument('--iter', default=500, type=int, help='Number of iterations')
    parser.add_argument('--dataset', default='T50', type=str, help='Dataset (T50 or movie)')
    parser.add_argument('--from_scratch', action='store_true', help='If not set, uses tagged_labeled CSVs')
    args = parser.parse_args()

    tagged_labeled_data_path = 'data/{}/tagged_labeled_data.csv'.format(args.dataset)

    if args.from_scratch or not exists(tagged_labeled_data_path):
        labeled_data = label_data("data/{}/qtask_all.csv".format(args.dataset))
        postagged_data = postag_data(labeled_data, tagged_labeled_data_path)
    else:
        print('Reading pre-generated pos-tagged data...')
        postagged_data = read_postagged_data(tagged_labeled_data_path)

    # TRAIN/TEST SPLIT
    kf = KFold(n_splits=5)
    kf_id = 1
    for train_indices, test_indices in kf.split(postagged_data):
        print('\n\n\nRUNNING FOLD #{}...'.format(kf_id))
        postagged_data = np.array(postagged_data, dtype=object)
        postagged_train, postagged_test = postagged_data[train_indices], postagged_data[test_indices]

        print(len(postagged_train), 'train /', len(postagged_test), 'test')

        train_sentence_indices=[doc[0] for doc in postagged_train]
        test_sentence_indices=[doc[0] for doc in postagged_test]

        print("Training indices:", train_sentence_indices[:10], '...')
        print("Test indices:    ", test_sentence_indices[:10], '...')
        print('\n\n\n')

        print('Extracting features...')
        X_train = [extract_features(doc[1:]) for doc in postagged_train]
        y_train = [get_labels(doc[1:]) for doc in postagged_train]
        X_test = [extract_features(doc[1:]) for doc in postagged_test]
        y_test = [get_labels(doc[1:]) for doc in postagged_test]

        X_train = [[train_sentence_indices[index], x_sentence] for index, x_sentence in enumerate(X_train)]
        y_train = [[train_sentence_indices[index], y_sentence] for index, y_sentence in enumerate(y_train)]
        X_test  = [[test_sentence_indices[index],  x_sentence] for index, x_sentence in enumerate(X_test)]
        y_test  = [[test_sentence_indices[index],  y_sentence] for index, y_sentence in enumerate(y_test)]

        print('Constructing CRF trainer...')
        trainer = pycrfsuite.Trainer(verbose=True)

        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq[1], yseq[1])

        trainer.set_params({
            'c1':1.0,
            'c2':1.0,
            'max_iterations': args.iter,
            'feature.possible_transitions': True
        })

        print('Training...')
        trainer.train(r"crf.model")
        tagger1=pycrfsuite.Tagger()
        tagger1.open("crf.model")

        y_pred=[]

        for xseq in X_test:
            y_pred_sentence = tagger1.tag(xseq[1])
            y_pred.append([xseq[0], y_pred_sentence])

        coordinate_true = get_coordinate(X_test, y_test)
        coordinate_pred = get_coordinate(X_test, y_pred)

        true_sentence = get_predict_content(postagged_test, y_test)
        true_sentence = turn_sentences_to_str(true_sentence)

        pred_sentence = get_predict_content(postagged_test, y_pred)
        pred_sentence = turn_sentences_to_str(pred_sentence)

        # WRITE TO FILES FOR ROUGE SCORE

        tq_path = 'data/{}/true_quotes_{}.txt'.format(args.dataset, kf_id)
        tq_file = open(tq_path, 'w')
        for doc in true_sentence:
            true_quot = doc[3]
            if true_quot in ["", ".", "...", "!"]:
                true_quot = "N/A"
            tq_file.write("{}\n".format(true_quot))
        tq_file.close()

        pq_path = 'data/{}/pred_quotes_{}.txt'.format(args.dataset, kf_id)
        pq_file = open(pq_path, 'w')
        for doc in pred_sentence:
            pred_quot = doc[3]
            if pred_quot in ["", ".", "...", "!"]:
                pred_quot = "N/A"
            pq_file.write("{}\n".format(pred_quot))
        pq_file.close()

        f1_mean, recall_mean, precision_mean, f1_score_all = evaluate(coordinate_true, coordinate_pred, true_sentence)
        f1_results = "precision :{:.5f}\t recall:{:.5f}\t f1_score:{:.5f}".format(precision_mean, recall_mean, f1_mean)
        print(f1_results)

        files_rouge = FilesRouge()
        rouge_results = files_rouge.get_scores(pq_path, tq_path, avg=True)
        print(rouge_results)

        results_file = open('data/{}/results_{}.txt'.format(args.dataset, kf_id), 'w')
        results_file.write("K-fold: {}\nIterations: {}\n{}\n{}\n".format(kf_id, args.iter, f1_results, rouge_results))
        results_file.close()
        kf_id += 1
