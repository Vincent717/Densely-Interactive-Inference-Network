from util.data_processing import LABEL_MAP
import util.parameters as params
FIXED_PARAMETERS, config = params.load_parameters()
import os
import pickle

# LABEL_MAP = {
#     "entailment": 0,
#     "neutral": 1,
#     "contradiction": 2,
#     "hidden": -1
# }


def evaluate_classifier(classifier, eval_set, batch_size, save_wrong_answer=False, q=False):
    """
    Function to get accuracy and cost of the model, evaluated on a chosen dataset.

    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    correct = 0
    genres, hypotheses, cost = classifier(eval_set)
    cost = cost / (len(eval_set) / batch_size)
    # full_batch = int(len(eval_set) / batch_size) * batch_size

    confusion_matrix = [[0,0,0] for i in range(3)]

    if isinstance(hypotheses, tuple):
        # when using logic rules, we want to check q_y_x too
        hypotheses, qyxs = hypotheses
    else:
        qyxs = 0

    # confusion matrix

    # label \ predict | entailment | neutral | contradiction
    # -------------------------------------------------------
    # entailment      |            |         |              
    # neutral         |            |         |               
    # contradiction   |            |         |               
    wrong_answer = []

    for i in range(hypotheses.shape[0]):
        hypothesis = hypotheses[i]
        label = eval_set[i]['label']
        if hypothesis == label:
            correct += 1 
        else:
            tmp = eval_set[i]
            tmp['predict_label'] = hypothesis
            wrong_answer.append(tmp)
        confusion_matrix[label][hypothesis] += 1 

    if q and qyxs != 0:
        q_correct = 0 
        q_confusion_matrix = [[0,0,0] for i in range(3)]
        for i in range(qyxs.shape[0]):
            hypothesis = qyxs[i]
            label = eval_set[i]['label']
            if hypothesis == label:
                q_correct += 1 
            # else:
            #     tmp = eval_set[i]
            #     tmp['predict_label'] = hypothesis
            #     wrong_answer.append(tmp)
            #q_confusion_matrix[label][hypothesis] += 1 

    if save_wrong_answer:
        wrong_answer_path = os.path.join(FIXED_PARAMETERS["log_path"], "wrong_answer.pkl")
        with open(wrong_answer_path, 'wb') as f:
            f.write(pickle.dumps(wrong_answer))
        print('wrong answer saved!') 

    confmx = """    label \ predict | entailment | neutral | contradiction
    -------------------------------------------------------
    entailment      |     {}     |    {}   |    {}        
    neutral         |     {}     |    {}   |    {}         
    contradiction   |     {}     |    {}   |    {}         """.format(\
        confusion_matrix[0][0],confusion_matrix[0][1],confusion_matrix[0][2],\
        confusion_matrix[1][0],confusion_matrix[1][1],confusion_matrix[1][2],\
        confusion_matrix[2][0],confusion_matrix[2][1],confusion_matrix[2][2])

    if q:
        return correct / float(hypotheses.shape[0]), q_correct / float(qyxs.shape[0]), cost, confmx
    else:
        return correct / float(hypotheses.shape[0]), cost, confmx

def evaluate_classifier_genre(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model by genre, evaluated on a chosen dataset. It returns a dictionary of accuracies by genre and cost for the full evaluation dataset.
    
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    genres, hypotheses, cost = classifier(eval_set)
    correct = dict((genre,0) for genre in set(genres))
    count = dict((genre,0) for genre in set(genres))
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        genre = genres[i]
        if hypothesis == eval_set[i]['label']:
            correct[genre] += 1.
        count[genre] += 1.

        if genre != eval_set[i]['genre']:
            print ('welp!')

    accuracy = {k: correct[k]/count[k] for k in correct}

    return accuracy, cost

