"""
Linear support vector classification model
"""

import argparse
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import utils as dp

def main():
    parser = argparse.ArgumentParser(description='Linear Support Vector Classification Model')
    parser.add_argument('--embedding',default='embedding/glove.6B.50d.subset.oov.vec', help='Path to the embedding')
    parser.add_argument('--train', default='data/train.tsv', help='Path to training data')
    parser.add_argument('--dev', default='data/dev.tsv', help='Path to dev data')
    parser.add_argument('--test', default='data/test.tsv', help='Path to test data')
    parser.add_argument('--predict', default='results/svm-subj-full.result', help='Path to the prediction file')
    
    args = parser.parse_args()
    embedding,embed_dim = dp.load_word2vec_embedding(args.embedding)

    X_train, y_train = dp.load_data(args.train, textindex=1, labelindex=0)
    X_dev, y_dev = dp.load_data(args.dev, textindex=1, labelindex=0)
    X_test, y_test = dp.load_data(args.test, textindex=1, labelindex=0)

    # Get index-word/label dicts for lookup:
    vocab_dict = dp.get_index_dict(X_train + X_dev + X_test)

    # Replace words / labels in the data by the according index
    vocab_dict_flipped = dict((v,k) for k,v in vocab_dict.items())

    # Get indexed data and labels
    X_train_index = [[vocab_dict_flipped[word] for word in chunk] for chunk in X_train]
    X_dev_index =  [[vocab_dict_flipped[word] for word in chunk] for chunk in X_dev]
    X_test_index =  [[vocab_dict_flipped[word] for word in chunk] for chunk in X_test]

    # Get embedding matrix:
    embed_matrix = dp.get_embedding_matrix(embedding,vocab_dict)

    # Use the simple count over all features in a single example:
    # Do average over word vectors:
    X_train_embedded = [np.mean([embed_matrix[element] for element in example], axis=0) for example in X_train_index]
    X_dev_embedded = [np.mean([embed_matrix[element] for element in example], axis=0) for example in X_dev_index]
    X_test_embedded = [np.mean([embed_matrix[element] for element in example], axis=0) for example in X_test_index]

    print("Loaded data.")

    # Tune C on the dev set, test on the test set:
    best_acc = 0.0
    best_c_acc = 0.0

    for c in [0.001, 0.01, 0.1, 1, 2, 4, 8, 16, 32, 64, 128, 256]:
        model_svr = SVC(C=c, kernel='linear', probability=True)
        model_svr.fit(X_train_embedded, y_train)

        # Use dev set to tune our hyperparameters
        pred_svr = model_svr.predict(X_dev_embedded)
        true_svr = y_dev

        acc = accuracy_score(true_svr,pred_svr)

        if acc > best_acc:
            best_acc = acc
            best_c_acc = c

    print("Best dev score: ", best_acc)

    # Test best model on test set
    best_model = SVC(C=best_c_acc, kernel='linear', probability=True)
    best_model.fit(X_train_embedded, y_train)

    best_pred = best_model.predict(X_test_embedded)

    test_acc = accuracy_score(y_test,best_pred)

    outlog = open(args.predict,'w')
    outlog.write('true\tpred\n')
    for true,pred in zip(y_test,best_pred):
        outlog.write('{}\t{}\n'.format(true, pred))
    outlog.close()

    print("Best C: ",best_c_acc)
    print("Test score: ",test_acc)

    print("Done")

if __name__ == '__main__':
    main()
