# word2vec_basic.py - script for training a word2vec model
# note:  large parts of this code are from the tensorflow word2vec tutorial (by Google)
# at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
# which is licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import absolute_import
from __future__ import print_function

import tensorflow.python.platform

import collections
import math
import numpy as np
import os
import random
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import zipfile
import argparse,time,pickle

parser = argparse.ArgumentParser(description='DBLP import script')
parser.add_argument('-f','--filename',required = True,
        help='Path and filename for dblp plain text file')
parser.add_argument('-o','--output_path',required = True,
        help='Path for output file(s)')
parser.add_argument('-v','--vocabulary_size',required = True,
        help='Number of entries in vocabulary', type=int)
parser.add_argument('-e','--embedding_size',required = True,
        help='Number of dimenstions of embedding', type=int)
parser.add_argument('-s','--skip_window',default=3,
        help='Number of dimenstions of embedding', type=int)
        
args = parser.parse_args()
filename=args.filename
word2vec_embedding_filename=args.output_path+"/word2vec-embedding.emb.pkl"


def save_data():
    print("saving data to file")
    f=open(word2vec_embedding_filename,"w")
    pickle.dump([data, count, dictionary, reverse_dictionary,final_embeddings],f)
    f.close()
    f=open(word2vec_embedding_filename+".info","w")
    f.write("trainingsteps:"+str(step)+"\n")
    f.write("last learning rate:"+str(current_learning_rate)+"\n")
    f.write("vocabulary_size:"+str(vocabulary_size)+"\n")
    f.write("embedding_size:"+str(embedding_size)+"\n")
    f.write("input data filename:"+str(filename)+"\n")
    f.write("skip_window:"+str(skip_window)+"\n")
    f.write("num_skips:"+str(num_skips)+"\n")
    f.write("num_sampled:"+str(num_sampled)+"\n")
    f.write("last_loss:"+str(last_loss)+"\n")
    f.close()
    print("done")

# Read the data into a string.
def read_data(filename):
  words=[]
  f = open(filename)
  for line in f:
    for word in line.split():
        words.append(word)
  f.close()
  return words

words = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = args.vocabulary_size

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], '->', labels[i, 0])
  print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = args.embedding_size  # Dimension of the embedding vector.
skip_window = args.skip_window       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(np.arange(valid_window), valid_size))
num_sampled = 64    # Number of negative examples to sample.
learning_rate = 1.0
current_learning_rate=learning_rate
graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                     num_sampled, vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  #optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
  _lr = tf.Variable(learning_rate, trainable=False)
  optimizer = tf.train.GradientDescentOptimizer(_lr).minimize(loss)
  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

# Step 5: Begin training.
num_steps = 50000001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  tf.initialize_all_variables().run()
  print("Initialized")

  average_loss=last_loss=average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 5000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      last_loss=average_loss
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 50000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
    if step>0 and step % 200000 == 0:
            tmp_start=time.time()
            final_embeddings = normalized_embeddings.eval()  
            save_data()
            print("saving took ",time.time()-tmp_start," seconds")
    if step>0 and step % 10000 == 0:
            tmp_start=time.time()
            current_learning_rate=learning_rate-(step*((0.9*learning_rate)/num_steps))
            session.run(tf.assign(_lr,current_learning_rate))
            print("current learning rate:",current_learning_rate)
            #print("changing LR took ",time.time()-tmp_start," seconds")
  final_embeddings = normalized_embeddings.eval()

save_data()

