"""
MLP Model for doing n-fold cross-validation on the data
Note, we represent documents with a simple mean of all word embeddings (ordering is lost)
"""

import argparse
import numpy as np

from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import Callback

from sklearn.metrics import accuracy_score

import data_processing as dp

def mlp_model(embedd_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu',input_shape=(embedd_dim, )))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='sigmoid'))
    return model

def main():
    parser = argparse.ArgumentParser(description='Multi-layer Perceptron for Text Classification')
    parser.add_argument('--embedding',default='embedding/glove.6B.50d.subset.oov.vec', help='Path to the embedding')
    parser.add_argument('--train', default='data/train.tsv', help='Path to input data')
    parser.add_argument('--dev', default='data/dev.tsv', help='Path to dev data')
    parser.add_argument('--test', default='data/test.tsv', help='Path to test data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=5,help='Batch size')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for weight initialization')
    parser.add_argument('--optimizer', default = 'adagrad', help='Optimizer to pick from. Supports all optimizers available in keras.')
    parser.add_argument('--model', default='results/mlp-full.model', help='Path to store the weights in.')
    
    args = parser.parse_args()
    # embed_dim equals the number of rows
    embedding,embed_dim = dp.load_word2vec_embedding(args.embedding)
    seed = args.random_seed
    model_path = args.model + '_' + str(seed) + '.model'

    # For keras and theano, it is ok to fix the numpy random seed.
    np.random.seed(seed)

    # Data format is:
    X_train, y_train = dp.load_data(args.train, textindex=1, labelindex=0)
    X_dev, y_dev = dp.load_data(args.dev, textindex=1, labelindex=0)
    X_test, y_test = dp.load_data(args.test, textindex=1, labelindex=0)

    # Get index-word/label dicts for lookup:
    vocab_dict = dp.get_index_dict(X_train + X_dev + X_test)
    label_dict = {'subjective':0, 'objective':1}

    # Replace words / labels in the data by the according index
    vocab_dict_flipped = dict((v,k) for k,v in vocab_dict.items())
    label_dict_flipped = {0:'subjective', 1:'objective'}

    # Get indexed data and labels
    X_train_index = [[vocab_dict_flipped[word] for word in chunk] for chunk in X_train]
    X_dev_index =  [[vocab_dict_flipped[word] for word in chunk] for chunk in X_dev]
    X_test_index =  [[vocab_dict_flipped[word] for word in chunk] for chunk in X_test]

    y_train_index = dp.get_binary_labels(label_dict, y_train)
    y_dev_index = dp.get_binary_labels(label_dict, y_dev)

    # Get embedding matrix:
    embed_matrix = dp.get_embedding_matrix(embedding,vocab_dict)

    # Use the simple count over all features in a single example:
    # Do average over word vectors:
    X_train_embedded = np.array([np.mean([embed_matrix[element] for element in example], axis=0) for example in X_train_index])
    X_dev_embedded = np.array([np.mean([embed_matrix[element] for element in example], axis=0) for example in X_dev_index])
    X_test_embedded = np.array([np.mean([embed_matrix[element] for element in example], axis=0) for example in X_test_index])

    weightsPath = model_path

    # Class for checking f1 measure during training
    class AccScore(Callback):
        def on_train_begin(self, logs={}):
            self.best_acc = 0.0
        def on_epoch_end(self, batch, logs={}):
            # Get predictions
            predict = np.asarray(self.model.predict(self.validation_data[0],batch_size=args.batch_size))
            # Flatten all outputs and remove padding
            pred = []
            true = []
            for doc_pred,doc_true in zip(predict,self.validation_data[1]):
                true.append(label_dict_flipped[doc_true.tolist().index(max(doc_true))])
                pred.append(label_dict_flipped[doc_pred.tolist().index(max(doc_pred))])
            self.accs=accuracy_score(pred, true)
            if self.accs > self.best_acc:
                self.best_acc=self.accs
                model.save_weights(weightsPath)
            return

    accscore_met= AccScore()

    model = mlp_model(embed_dim)
    model.compile(args.optimizer, 'binary_crossentropy',metrics=[metrics.categorical_accuracy])
    
    model.fit(X_train_embedded, y_train_index, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_dev_embedded, y_dev_index), verbose=1, callbacks=[accscore_met])

    # Get the best model for one fold and do prediction:
    model.load_weights(weightsPath)
    result = model.predict(X_test_embedded)

    pred = []
    for i in range(len(result)):
        pred.append(label_dict_flipped[result[i].tolist().index(max(result[i]))])

    print("Test accuracy: ",accuracy_score(pred, y_test))
    print("Done")

if __name__ == '__main__':
    main()
