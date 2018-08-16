"""Class handling helper functions for data processing

Offers functions for reading data and embeddings. 
Also some simple utility functions for reading prediction files,
and for generating embedding matrices
"""
import numpy as np

def load_word2vec_embedding(embedding):
    """Load word2vec embedding 

    @param: embedding -- path to the embedding
    @returns: word2vec -- the embedding dict
    @returns: embedd_dim -- the dimension of the embedding
    """
    model = {'dimension':0, 'vocab':set(), 'vectors':dict()}
    with open(embedding) as lines:
        firstline = True
        for line in lines:
            if firstline:
                firstline = False
                model['dimension']=int(line.strip().split()[-1])
                continue
            splits = line.strip().split()
            # Add word and its vector to the embedding
            model['vocab'].add(splits[0])
            model['vectors'][splits[0]] = np.fromstring(' '.join(splits[1:]), dtype=float, sep=' ')
    embedd_dim = model['dimension']
    return model, embedd_dim

def load_data(input_data, textindex=4, labelindex=3):
    """Load tsv format data

    @param: input_data -- path to the data 
    @returns: data -- the data : one list of words in the document
    @returns: classes -- the class labels
    """
    labels = []
    data = []
    with open(input_data) as lines:
        next(lines) # Skip the first line of a tsv document, as it is the header
        for line in lines:
            try:
                line.strip().split('\t')[textindex]
            except IndexError:
                print("Error with following line:")
                print(line)
            # Data is already tokenized, so a simple split is sufficient
            tokens = line.strip().split('\t')[textindex].split()
            cleaned_tokens = [''.join(tok.split()) for tok in tokens]
            data.append(cleaned_tokens)
            labels.append(line.strip().split('\t')[labelindex])
    return data,labels

def get_index_dict(input_data):
    """Create index - word/label dict from list input

    @params : List of lists
    @returns : Index - Word/label dictionary
    """
    result = dict()
    vocab = set()
    i = 1
    # Flatten list and get indices
    for element in [word for document in input_data for word in document]:
        if element not in vocab:
            result[i]=element
            i+=1
            vocab.add(element)
    return result

def get_flipped_dict(dictionary):
    """Simple function to get a dictionary where the keys and values are flipped
    """
    return dict((v,k) for k,v in dictionary.items())

def get_label_dict(input_data):
    """Create index - label dict from list input

    @params : List of lists
    @returns : Index - Word/label dictionary
    """
    result = dict()
    vocab = set()
    i = 0
    # Flatten list and get indices
    for element in sorted(input_data):
        if element not in vocab:
            result[i]=element
            i+=1
            vocab.add(element)
    return result

def get_binary_labels(label_dict, labels):
    """Function to get index labels for binary classification.
    Labels are either [0, 1] or [1, 0].
    """
    result = []
    for label in labels:
        label_vec = [0, 0]
        label_vec[label_dict[label]]=1
        result.append(label_vec)
    return np.array(result)
    

def get_embedding_matrix(embedding, vocab_dict):
    """Get the embedding matrix containing the weight to the according index of the vocabulary
    Note: This function ignores OOV words, it is assumed that the embeddings contains vectors for all OOV words.
    Common practices to handle OOV words:
     * Map everything to an UNK token
     * Initialize random vectors for each OOV word
     * Map them to zero (don't do that)

    @params : embedding - word2vec formatted embedding
    @params : vocab_dict - index-word dictionary of the vocabulary
    @ returns : e_mat - The embedding matrix for the words in the vocabulary (OOV words are zero)
    """
    total_vocab = set(list(embedding['vocab']) + list(vocab_dict.values()))
    dimension = embedding['dimension']
    e_mat = np.zeros((len(vocab_dict) + 1, dimension))
    for index,word in vocab_dict.items():
        try:
            e_mat[index]=embedding['vectors'][word]
        except KeyError:
            continue
            # You may handle random initializations of out of vocabulary words here, however, for reproduceable results, you should do this in some preprocessing step:
            #wordvec = np.random.rand(dimension)
            #e_mat[index]= wordvec
    return e_mat

def load_vectors(embedding_matrix, X_data, y_data, padding_size, nb_classes):
    """Function to load x and y vectors from an embedding matrix and return post padded data.
    y data gets (always!) transformed to categorical for multi class classification problems.
    """
    # Transform the data using the embedding matrix
    X_data_embedded = np.array([embed_matrix[x] for x in X_data])
    # Do padding for all inputs:
    X_data_padded = sequence.pad_sequences(X_data_embedded,maxlen=padding_size, padding='post')
    # For categorical cross_entropy we need matrices representing the classes:
    # Note, that we pad after doing the transformation into the matrix!
    y_data_padded = sequence.pad_sequences(np.asarray([np_utils.to_categorical(y_label,nb_classes+1) for y_label in y_data]),maxlen=padding_size, padding='post')
    return X_data_padded, y_data_padded

def get_array_index(bigarray, searcharray):
    """Search vectors in numpy arrays and returns the index.
    If the vector was not found, return "Not found."
    """
    index = 0
    for elem in bigarray:
        if np.array_equal(searcharray, elem):
            return index
        else:
            index += 1
    return "Not found."

def read_active_learning_history(infile):
    """Function for reading an active learning history file.
    Those files have the following format: 
     Iteration\tScore\n
    Returns a list of (iteration,score) pairs
    """
    result = []
    with open(infile) as lines:
        next(lines) # Ignore first line
        for line in lines:
            result.append((line.strip().split('\t')[0],line.strip().split('\t')[-1]))
    return result
 

