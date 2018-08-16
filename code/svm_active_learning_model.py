"""
Linear SVM with Active Learning
"""

import argparse
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

import data_processing as dp
import visualize as vz
from active_learning import Active_Learning


def main():
    parser = argparse.ArgumentParser(description='Support Vector Classification Model')
    parser.add_argument('--embedding',default='embedding/glove.6B.50d.subset.oov.vec', help='Path to the embedding')
    parser.add_argument('--train', default='data/train_labeled.tsv', help='Path to training data')
    parser.add_argument('--unlabeled', default='data/train_unlabeled.tsv', help='data to sample from')
    parser.add_argument('--test', default='data/test.tsv', help='Path to test data')
    parser.add_argument('--sampling', default='random', help='active learning heuristic for uncertainty sampling')
    parser.add_argument('--predict', default='results/svm-al-random.result', help='Path to the prediction file')
    parser.add_argument('--al_history', default='results/svm-al-random.history', help='Our active learning history')
    parser.add_argument('--max_iter',type=int, default=500, help='Maximal number of active learning iterations')
    parser.add_argument('--c', type=float, default=2, help='C parameter for SVM')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for random sampling')
    
    args = parser.parse_args()
    embedding,embed_dim = dp.load_word2vec_embedding(args.embedding)
    predict_file = args.predict 
    c = args.c

    # Data format is:
    # Pre-word \t Gap-word \t Suc-word \t Error
    X_train, y_train = dp.load_data(args.train, textindex=1, labelindex=0)
    X_test, y_test = dp.load_data(args.test, textindex=1, labelindex=0)

    # Active learning data
    X_active, y_active = dp.load_data(args.unlabeled, textindex=1, labelindex=0)

    # Get index-word/label dicts for lookup:
    vocab_dict = dp.get_index_dict(X_train + X_test + X_active)

    # Replace words / labels in the data by the according index
    vocab_dict_flipped = dict((v,k) for k,v in vocab_dict.items())

    # Get indexed data and labels
    X_train_index = [[vocab_dict_flipped[word] for word in chunk] for chunk in X_train]
    X_test_index =  [[vocab_dict_flipped[word] for word in chunk] for chunk in X_test]

    # Active learning data
    X_active_index =  [[vocab_dict_flipped[word] for word in chunk] for chunk in X_active]

    print ("Number of initial training documents: ",len(X_train))

    # Get embedding matrix:
    embed_matrix = dp.get_embedding_matrix(embedding,vocab_dict)

    # Use the simple count over all features in a single example:
    # Do average over word vectors:
    X_train_embedded = [np.mean([embed_matrix[element] for element in example], axis=0) for example in X_train_index]
    X_test_embedded = [np.mean([embed_matrix[element] for element in example], axis=0) for example in X_test_index]

    # Active learning
    X_active_embedded = [np.mean([embed_matrix[element] for element in example], axis=0) for example in X_active_index]

    # Do active learning as long as there is data:
    pool_data = X_active_embedded[:]
    pool_labels = y_active[:]

    # Active learning results for visualization
    step, acc = [],[]

    print("Loaded data.")

    iteration = 0
    outlog = open(args.predict,'w')
    outlog.write('Iteration\tC\tAcc\n')

    activelog = open(args.al_history,'w')
    activelog.write('Iteration\tLabel\tText\n')

    while len(pool_data) > 1 and iteration < args.max_iter:
        if len(X_train_embedded) % 50 == 0:
            print("Training on: ", len(X_train_embedded), " instances.")

        model_svm = SVC(C=c, kernel='linear', probability=True)
        model_svm.fit(X_train_embedded, y_train)

        pred = model_svm.predict(X_test_embedded)

        test_acc = accuracy_score(y_test,pred)

        outlog.write('{}\t{}\t{}\n'.format(iteration,c,test_acc))
        step.append(iteration); acc.append(test_acc)

        # Add data from the pool to the training set based on our active learning:
        al = Active_Learning(pool_data,model_svm, args.seed)
        if args.sampling == 'random':
            add_sample_data = al.get_random()
        else:
            add_sample_data = al.get_most_uncertain(args.sampling)

        # Get the data index from pool
        sample_index = dp.get_array_index(pool_data, add_sample_data)
        
        # Get the according label
        add_sample_label = pool_labels[sample_index]

        # Add the results to our learning history
        activelog.write('{}\t{}\t{}\n'.format(iteration, add_sample_label, ' '.join(X_active[sample_index])))

        # Add it to the training pool
        X_train_embedded.append(add_sample_data)
        y_train.append(add_sample_label)

        # Remove labeled data from pool
        del pool_labels[sample_index]
        del pool_data[sample_index]

        iteration += 1

    outlog.close()

    vz.plot(step, acc)

    print("Done")

if __name__ == '__main__':
    main()
