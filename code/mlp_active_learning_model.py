"""
Linear SVM with Active Learning
"""
import argparse
import numpy as np

from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import Callback

from sklearn.metrics import accuracy_score

import data_processing as dp
from active_learning import Active_Learning

def main():
    parser = argparse.ArgumentParser(description='Multi-layer Perceptron for Text Classification')
    parser.add_argument('--embedding',default='embedding/glove.6B.50d.subset.oov.vec', help='Path to the embedding')
    parser.add_argument('--train_labeled', default='data/train_labeled.tsv', help='Path to training data')
    parser.add_argument('--dev', default='data/dev.tsv', help='Path to dev data')
    parser.add_argument('--test', default='data/test.tsv', help='Path to test data')
    parser.add_argument('--train_unlabeled', default='data/train_unlabeled.tsv', help='data to sample from')
    parser.add_argument('--sampling', default='random', help='active learning heuristic for uncertainty sampling')
    parser.add_argument('--model', default='results/mlp-al', help='Path to the prediction file')
    parser.add_argument('--maximum_iterations',type=int, default=5, help='Maximal number of iterations')
    parser.add_argument('--random_sampling_seed', type=int, default=42, help='Python random seed for random sampling')
    parser.add_argument('--active_learning_history', default='results/mlp-al-random.result')
    parser.add_argument('--epochs',type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for one backward pass.')
    parser.add_argument('--neural_network_random_seed', type=int, default=42, help='random seed for numpy')
    parser.add_argument('--optimizer', default='adagrad', help='SGD optimizer')
    
    args = parser.parse_args()
    # Add the random seed to our model path
    model_path = args.model + '-' + str(args.neural_network_random_seed) + '.model'
    # For keras and theano, it is ok to fix the numpy random seed.
    np.random.seed(args.neural_network_random_seed)
    weights_path = model_path

    ###########################################
    #       Loading data and vectors
    ###########################################
    
    embedding,embed_dim = dp.load_word2vec_embedding(args.embedding)

    X_train, y_train = dp.load_data(args.train_labeled, textindex=1, labelindex=0)
    X_dev, y_dev = dp.load_data(args.dev, textindex=1, labelindex=0)
    X_test, y_test = dp.load_data(args.test, textindex=1, labelindex=0)

    # Active learning data
    X_active, y_active = dp.load_data(args.train_unlabeled, textindex=1, labelindex=0)

    # Get index-word/label dicts for lookup:
    # NOTE: Creating a dictionary out of all data has the implicit assumption 
    #       that all the words we encounter during sampling and testing we have already seen during training.
    vocab_dict = dp.get_index_dict(X_train + X_test + X_active) 
    label_dict = {'subjective':0, 'objective':1}

    # Replace words / labels in the data by the according index
    vocab_dict_flipped = dict((v,k) for k,v in vocab_dict.items())
    label_dict_flipped = {0:'subjective', 1:'objective'}

    # Get indexed data and labels
    X_train_index = [[vocab_dict_flipped[word] for word in chunk] for chunk in X_train]
    X_dev_index = [[vocab_dict_flipped[word] for word in chunk] for chunk in X_dev]
    X_test_index =  [[vocab_dict_flipped[word] for word in chunk] for chunk in X_test]

    # Active learning data
    X_active_index =  [[vocab_dict_flipped[word] for word in chunk] for chunk in X_active]

    print ("Number of initial training documents: ",len(X_train))

    # Get embedding matrix:
    embed_matrix = dp.get_embedding_matrix(embedding,vocab_dict)

    # Use the simple count over all features in a single example:
    # Do average over word vectors:
    X_train_embedded = np.array([np.mean([embed_matrix[element] for element in example], axis=0) for example in X_train_index])
    X_dev_embedded = np.array([np.mean([embed_matrix[element] for element in example], axis=0) for example in X_dev_index])
    X_test_embedded = np.array([np.mean([embed_matrix[element] for element in example], axis=0) for example in X_test_index])

    # Active learning
    X_active_embedded = np.array([np.mean([embed_matrix[element] for element in example], axis=0) for example in X_active_index])

    y_train_index = dp.get_binary_labels(label_dict, y_train)
    y_dev_index = dp.get_binary_labels(label_dict, y_dev)
    y_active_index = dp.get_binary_labels(label_dict, y_active)

    # Define our pools for active learning
    pool_data = X_active_embedded[:]
    pool_labels = y_active_index[:]

    print("Loaded data.")

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
                model.save_weights(weights_path)
            return

    accscore_met= AccScore()

    ###########################################
    #       Implement model
    ###########################################

    model = Sequential()
    # A simple dense layer with 128 hidden units. The activation function is ReLU.
    model.add(Dense(128, activation='relu',input_shape=(embed_dim, ))) 
    # Dropout acts as a regularizer to prevent overfitting.
    model.add(Dropout(0.4)) 
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    # The final layer does the predcition. 
    # Sigmoid is a common activation function for binary classifcation.
    model.add(Dense(2, activation='sigmoid')) 

    ###########################################
    #       Compile the model
    ###########################################
    # We first have to compile the model
    model.compile(args.optimizer, 'binary_crossentropy',metrics=[metrics.categorical_accuracy])

    ###########################################
    #       Start active learning
    ###########################################
    # Active learning results for visualization
    step, acc = [],[]

    iteration = 0

    outlog = open(args.active_learning_history,'w')
    outlog.write('Iteration\tAccuracy\n')

    while len(pool_data) > 1 and iteration < args.maximum_iterations:
        if len(X_train_embedded) % 50 == 0:
            print("Training on: ", len(X_train_embedded), " instances.")

        model.fit(X_train_embedded, y_train_index, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_dev_embedded, y_dev_index), verbose=0, callbacks=[accscore_met])

        # Load best weights and compute test performance
        model.load_weights(weights_path)
        result = model.predict(X_test_embedded)
        pred = []
        for i in range(len(result)):
            pred.append(label_dict_flipped[result[i].tolist().index(max(result[i]))])
        test_acc = accuracy_score(y_test,pred)
        outlog.write('{}\t{}\n'.format(iteration,test_acc))
        step.append(iteration); acc.append(test_acc)

        # Add data from the pool to the training set based on our active learning:
        al = Active_Learning(pool_data, model, args.random_sampling_seed)
        if args.sampling == 'random':
            add_sample_data = al.get_random()
        else:
            add_sample_data = al.get_most_uncertain(args.sampling)

        # Get the data index from pool
        sample_index = dp.get_array_index(pool_data, add_sample_data)
        
        # Get the according label
        add_sample_label = pool_labels[sample_index]

        # Add it to the training pool
        X_train_embedded = np.vstack((X_train_embedded, add_sample_data))
        y_train_index = np.vstack((y_train_index, add_sample_label))
    
        # Remove labeled data from pool
        pool_labels = np.delete(pool_labels, sample_index, axis=0) 
        pool_data = np.delete(pool_data, sample_index, axis=0)

        iteration += 1

    outlog.close()

if __name__ == '__main__':
    main()
