# What will you publish next?
Contents
 1. Overview
 2. Fun facts
 3. How it is done
 4. How you can use it

# 1. Overview

Recurrent neural networks (RNNs) are great tools, and data sets are plenty.
So why not ask an RNN, what you, with your outstanding publication record as a scientist, will publish next?

Is this really possible? Well, obviously not, but trying is insightful and fun!

## Spare me the details, where can I try it?
For the time being, we set up a web service, so you can try it out for yourself:

 [http://wi.bwl.uni-mainz.de/what-will-you-publish-next/](http://wi.bwl.uni-mainz.de/what-will-you-publish-next/)

Please go easy on it, the backend is not a big box, and the implementation is rather ad-hoc. You might find that this service is not working at times, and might stop working at some point altogether :-/

## The approach in a nutshell

So the task is to predict which paper someone will publish next, given previous papers of the same author. How do we get there?

- Dataset: we use the [DBLP](http://dblp.uni-trier.de/) [data set](http://dblp.uni-trier.de/xml/), which is a 1.7 GB XML monstrosity containing more than 3 million publications and 1.6 million authors in the area of computer science. Thanks to the DBLP team for making this data set public!
- We preprocess this data set and produce a ~1 GB plain text file containing paper titles and journal/conference names (see below how this looks like)
- Then, we train a [word2vec](https://en.wikipedia.org/wiki/Word2vec) model to embed the words into 256-dimensional vectors, so we can work with words rather than characters in the RNN
- Finally, we train an RNN, or, to be more precise, an [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) network to predict the next word, given the previous ones.

Now, to find out what someone will publish next, we

- retrieve some of his/her papers from the data set
- "prime" the network on these papers 
- and let it fantasize the next publications, word by word

See below for more details!
# 2. Fun facts
 TODO, some interesting results
# 3. How it is done
Now let us look at some of the gory details (although we will remain pretty high-level here).

## The data

The DBLP data comes in a large XML file, with a lot of information that we don't need. An example entry looks like:
```xml
...
<inproceedings mdate="2012-09-18" key="persons/Codd74">
<author>E. F. Codd</author>
<title>Seven Steps to Rendezvous with the Casual User.</title>
<month>January</month>
<year>1974</year>
<pages>179-200</pages>
<booktitle>IFIP Working Conference Data Base Management</booktitle>
<url>db/conf/ds/dbm74.html#Codd74</url>
<note>IBM Research Report RJ 1333, San Jose, California</note>
<cdrom>DS/DS1974/P179.pdf</cdrom>
</inproceedings>
...

```
We read in the data, and remember just the author(s), title, journal/conference name (or the publisher, in case of books), and the year.
We store the information in a Python pickle file for later use.

Then, we produce the output format, which looks like 
```
--beginentry-- seven steps to rendezvous with the casual user ;  in:  ifip working conference data base management
```

for the same publication we saw above. To increase the ease of handling, and reduce the size of the dictionary, we remove a lot of special characters and numbers, and use lower case letters only.
The special `--beginentry--` marks the beginning of a new publication. It is not strictly necessary (the RNN actually learns to produce well-formed output without it), but it makes things easier for us later on, when we display the data.

We create the file with the training data by iterating through all authors. For each author we put all his/her publications in the file. Hence, the file is redundant, but that is OK. What we assume is that the title and journal/conference of a publication depends on the other publications of the same author. Hence, each paper has to be in the list of papers of all its authors. 

This process yields a 1 GB training file. Ouch!

A side note: We are not trying to predict the author names, or the years. The reason is that authors are very hard to predict. There are 1,6 million authors in DBLP, so remembering a decent number of names already stretches the dictionary size. Also, the task is simply too hard: without the authors, the model "only" has to learn to produce well-formed sentences which somehow cover the same topics. Predicting authors means learning to correlate individuals with topics. First, this is difficult in itself because of the sheer numbers involved. Second, each individual author occurs only a few times (say, 100 times) in a 1GB data file. Picking this up this kind of weak correlation is very difficult. Third, the representation we chose uses only last names. So all authors named "Miller" are thrown together, which is clearly non-nonsensical if you try to link names to topics.
As for the years: This actually works fine, the model learns to sample papers in ascending year order if they are part of the data. But, of course, it has no concept of years in the future, and so this doesn't help us much (if you see that your "next" paper will be from 2010, you won't believe it). We decided to go without years.

## Word2vec
[Word2vec](https://en.wikipedia.org/wiki/Word2vec) is a great technique for embedding words in high dimensional vectors. The underlying idea is that words stored in a dictionary are pretty unhandy when it comes to using them in models. The dictionaries are large (tens of thousands of words), and syntactic or semantic relationships between words are not properly represented in a dictionary. 
Word2vec now calculates an *embedding*, that is, it changes the way the words are represented. Before, the are an index in a, afterwards, each word is represented by an n-dimensional vector of real numbers. Check out [this tutorial](https://www.tensorflow.org/versions/r0.7/tutorials/word2vec/index.html) on word2vec and embeddings to understand what is going on within the model (and witness some amazing properties of the embedding).
## RNNs
Before we start, if you haven't read Andrej Karpathy's [great blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) about RNNs, you should definitly do so. It got me started on RNNs (although I really liked Ilya Sutskever's [RNN demo](http://www.cs.toronto.edu/~ilya/rnn.html) at the time).
RNNs are machine learning models for sequences. Sequence learning is a very common task. You might want to predict the weather of tomorrow, given today's weather, predict the next letter of a text, given previous letters, or the next word of a sentence, given previous words.

RNNs can learn to model sequences. As neural networks, they consist of neurons connected by weights (synapses, connections), which are used to pass information around. A neuron collects the incoming information, and then calculates its own activation using a simple, often non-linear function. Usually, the neurons form "layers", i.e. groups of neurons which are connected to neurons in other layers, but not to neurons within the same layer (see left-hand side of image below). 

An RNN now usually contains multiple layers (see middle of figure below). The input layer is used to feed data to the network (i.e. the last known character). A "hidden" layer gets information from the input, but we don't explicitly know what the neurons in there are modeling.
For the kind of RNNs we're dealing with, the output layer contains the "prediction", i.e., tomorrows weather or the next word in the sequence.
The "sequence" part of the model are the *recurrent* connections in the hidden layer (the arrow that points from the hidden layer back to itself). They pass information form one step of the sequence to the next one. 

![RNN](http://wi.bwl.uni-mainz.de/what-will-you-publish-next/rnn.jpg)

This is much easier to understand if we "unroll" the RNN for demonstration purposes (see right hand side). Now, each step in the sequence is represented by its "own" input, hidden, and output layer. In the image, the current "state" of the network is that it has received input in three preceding periods (letters "h", "e", and "l"), and now receives the input "l". Its task is to correctly predict the next letter in the output layer.

We can measure how good the network does its job by calculating the error it makes (as we know the true output). **Learning** is now performed using the [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) algorithm. That is, we calculate the error, and calculate all gradients of the parameters (weights, connections) in the network. In other words, we calculate in which direction we need to change the weights, in order to reduce the error. We can do this, because all functions  in the model are differentiable. 

To **sample** new data from the model, we can now ask it to fantasize new outputs. We can "prime" the network by initializing its past state to something we want to, in this case, for example the letters "h" "o" "u" and "s". If the network has been trained on English words/texts it is very likely that it will deem "e" to be a probable next character, to write the word "house". Or it might sample an "i" and continue to write the word "housing".

We use LSTM neurons, which are slightly more complicated in their inner workings, but enable the RNN to learn long-term relationships more easily.

Note that, in our example, we operate on words instead of letters. This makes the RNN pick up temporal relations that extend further into the "past" of the sequence (i.e., 10 words into the "past" is far more than 10 letters).

## The technical stuff
We use

- Python for preprocessing
- [Tensorflow](http://www.tensorflow.org) for the word2vec and LSTM/RNN models (in fact, we use some of the tensorflow tutorial code)
 (we used tensorflow 0.6, it seems that it does not work with tf 0.7, since they changed some indexing features)
- Ubuntu 14.04
- An Nvidia GTX 970 GPU for training the LSTM model

# 4. How you can use it

Download the dblp data and unzip it

```bash
wget http://dblp.uni-trier.de/xml/dblp.xml.gz
```

Preprocesses the XML file (sometimes, closing tags come before opening tags - this confuses our script)

```bash
cat dblp.xml|sed "s/><article/>\n<article/g"|sed "s/><inproceedings/>\n<inproceedings/g"|sed "s/><proceedings/>\n<proceedings/g"|sed "s/><incollection/>\n<incollection/g"|sed "s/><book/>\n<book/g">dblp-cleaned.xml
```

Import the data, an create the training data file (can take a couple of minutes)

```bash
python dblp_import.py -f dblp-cleaned.xml -o <output_dir>
```

Train the word2vec model (let this run for... 1-2 days, or at least a couple of hours)
```bash
python word2vec_basic.py -f dblp-plaintext.txt -o  <output_path> -v <dictionary_size> -e <embedding_size>
```
we use a dictionary size of 75k, and 256 dimensions for the embedding. But try smaller values, it should work OK as well, and train much faster.

Then, train the LSTM on the data (let this run for... 2-3 days, or at least half a day)
```bash
python lstm.py -f word2vec-embedding.emb.pkl  -o <output_path> -l <time_steps>
```
where `<time_steps>` is the number of steps that the RNN will look "into the past". We use 50 time steps to make sure the previous paper is still "in memory" when sampling a new one. 

Finally, when the model is trained, we can sample from it 
```bash
python lstm.py -f word2vec-embedding.emb.pkl  -o <output_path> -l <time_steps> -s True -n <number_words> -p <prime_text>
```
where the script expects to find the file `<output_path>/lstm-model.ckpt`, and the network will sample `<number_words>` words. `<prime_text>` is the text that you want your sample to be dependent on, i.e., the past publications of the author in question. For example, `<prime_text>` could be `the joy of cheesecake ; in: cooking heroes`. Note that the words of the prime should be present in the dicitionary that was created in the word2vec step (contained in the embedding file).

You can also instantiate your own LSTM instance from the LSTM class in lstm.py, this will give you access to the sampled data and the underlying probabilities - the script just prints the data on the terminal.

