# lstm.py - Class for training and sampling a Recurrent Neural Network with LSTM units
# note:  parts of this code are inspired by the tensorflow LSTM tutorial at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py

import time,argparse,sys
import tensorflow as tf
import numpy as np
import cPickle as pickle
from tensorflow.models.rnn import rnn_cell
import re,inspect,time


# some static methods
def print_box(vars_to_print):
    print "-----------------------------"
    for name,var in vars_to_print:
        print "|\t",name,"\t",var
    
def load_embeddings(filename):
    # load created embedding
    print "loading word embedding...",
    sys.stdout.flush()
    with open(filename) as f:
        data, count, dictionary, reverse_dictionary, saved_embedding=pickle.load(f)
    print "done"
    print "Data, first 30 words"
    print [reverse_dictionary[data[x]] for x in range(30)]

    print ""
    
    return data, count, dictionary, reverse_dictionary, saved_embedding

class LSTM():
    # initialize the LSTM
    def __init__(self,count, dictionary, reverse_dictionary, saved_embedding,model_path,num_sampling_steps=20,data=False,sample_model=False,num_steps=50,batch_size=24):
        # lstm config
        self.embedding_dims=saved_embedding.shape[1]
        self.lstm_size=self.embedding_dims
        self.dictionary=dictionary
        self.reverse_dictionary=reverse_dictionary
        self.saved_embedding=saved_embedding
        self.data=data
        self.vocab_size=len(dictionary)    
        self.model_path=model_path
        self.only_for_sampling=sample_model
        if sample_model:
            self.batch_size=1
        else:
            self.batch_size=batch_size
        self.num_steps=num_steps
        self.num_layers=2
        self.max_epochs=10
        self.lr_decay=0.8
        self.learning_rate=1.0
        self.max_grad_norm=5
        self.num_sampling_steps=num_sampling_steps
        
        print "Configuration:"
        print_box([("lstm_size",self.lstm_size),("embedding_dims",self.embedding_dims),("sample_model",sample_model),("self.batch_size",self.batch_size),("self.num_steps",self.num_steps),("self.num_layers",self.num_layers),('self.lr_decay',self.lr_decay),('self.max_epochs',self.max_epochs),('self.learning_rate',self.learning_rate),('self.max_grad_norm',self.max_grad_norm)])

        self.create_model()
    
    # set up the model by creating the appropriate tensorflow variables    
    def create_model(self):
        print "Setting up model",
        sys.stdout.flush()
        # placeholders for data + targets
        self._input_data = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps))
        self._targets = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])

        # set up lookup function
        self.embedding = tf.constant(self.saved_embedding,name="embedding")
        self.inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)
        # lstm model
        self.lstm_cell = rnn_cell.BasicLSTMCell(self.lstm_size)
        self.cell = rnn_cell.MultiRNNCell([self.lstm_cell] * self.num_layers)


        self._initial_state = self.cell.zero_state(self.batch_size, tf.float32)

        from tensorflow.models.rnn import rnn
        self.inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, self.num_steps, self.inputs)]
        self.outputs, self.states = rnn.rnn(self.cell, self.inputs, initial_state=self._initial_state)

        self.output = tf.reshape(tf.concat(1, self.outputs), [-1, self.lstm_size])
        self.softmax_w = tf.get_variable("softmax_w", [self.lstm_size, self.vocab_size])
        self.softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
        self.logits = tf.matmul(self.output, self.softmax_w) + self.softmax_b
        self._final_state = self.states[-1]
        self.saver = tf.train.Saver()
        
        #delete data to save memory if network is used for sampling only
        if self.only_for_sampling:
            del self.data
            
        print "done"
    
    # sample a trainied model, given some prime text (and, potentially, a temperature for the softmax)
    def sample_model(self,prime_text=None,sampling_temp=1.0):
        output_words=[]
        output_probs=[]
        prime_words=[]
        print "---- Sampling model ----"
        with tf.device('/cpu:0'):
            with tf.Session() as session:
                #print "_------------------"
                #print self.model_path+"/lstm-model.ckpt"
                #print "_------------------"
                tmp_start=time.time()
                self.saver.restore(session, self.model_path+"/lstm-model.ckpt")
                state = self._initial_state.eval()
                print "loading took %f seconds"%(time.time()-tmp_start)
                # set initial x (self.batch_size is 1)
                if prime_text==None:
                    x=np.random.randint(0,self.vocab_size,size=(self.batch_size, self.num_steps))
                else:
                    x=np.zeros([self.batch_size, self.num_steps],dtype=np.int32)
                    split_text=prime_text.split()
                    #print split_text
                    for i in range(self.num_steps):
                        txt=split_text[i%len(split_text)]
                        if txt in self.dictionary.keys():
                            x[0,i]=self.dictionary[txt]
                        else:
                            x[0,i]=0
                print "priming:", 
                for i in range(self.num_steps):
                    print self.reverse_dictionary[x[0,i]],
                    prime_words.append(self.reverse_dictionary[x[0,i]])
                delayed_linebreak=False
                print "...\n"
                for step in range(self.num_sampling_steps):
                    #print "step:",step
                    #print x
                    out,state, _ = session.run([self.logits,self._final_state, tf.no_op()],
                                                     {self._input_data: x,
                                                      self._initial_state: state})
                    #out[-1][0]=0
                    
                    # sample from output distribution
                    e=np.exp(out[-1]/sampling_temp)
                    probs=e/np.sum(e)
                    probsum=np.sum(probs)
                    # if dictionary is very large, rounding errors may yield a sum > 1
                    if probsum>=0.9999:
                        probs=probs/(1.002)
                    # sample the distribution!
                    y=np.argmax(np.random.multinomial(1,probs))
                    output_probs.append(probs[y])    
                    
                    x[:,0:-1]=x[:,1:]
                    x[:,-1]=y
                    
                    # retrieve the actual string
                    outstring=self.reverse_dictionary[y]
                    output_words.append(outstring)
                    if outstring=="--beginentry--":
                        print ""
                    print outstring,
                    """
                    if outstring=="--endentry--" or outstring=="--endauthor--":
                        print ""
                    if delayed_linebreak:
                        print ""
                        delayed_linebreak=False
                    if outstring=="--beginauthor--":
                        delayed_linebreak=True
                    """
                    sys.stdout.flush()
                    #print state
                    #print state[-1]
                print "\ndone"
        return output_words,output_probs,prime_words
    
    # train the model
    def train_model(self,reload_checkpoint=False):
        #set up loss function and cost
        loss = tf.models.rnn.seq2seq.sequence_loss_by_example([self.logits],
                                            [tf.reshape(self._targets, [-1])],
                                            [tf.ones([self.batch_size * self.num_steps])],
                                            self.vocab_size)

        _cost = cost = tf.reduce_sum(loss) / self.batch_size

        # learning rate shall be variable, not fixed
        _lr = tf.Variable(self.learning_rate, trainable=False)
        
        # get tensorflow gradients 
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                              self.max_grad_norm)
        # istantiate optimizer and tensorflow training operation
        optimizer = tf.train.GradientDescentOptimizer(_lr)
        _train_op = optimizer.apply_gradients(zip(grads, tvars))

        print "---- Training model ----"
        # we use the ptb reader
        from tensorflow.models.rnn.ptb import reader
        self.saver = tf.train.Saver()
        # open tensorflow session and start training 
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95))) as session:
            init=tf.initialize_all_variables()
            session.run(init)
            # enables the model to continue from a previously saved checkpoint
            if not reload_checkpoint:

                i_start=0
                step_start=0
            else:
                print "restoring model"
                self.saver.restore(session, self.model_path+"/lstm-model.ckpt")
                state = self._initial_state.eval()
                print "restoring training state"
                i_start,step_start,costs_start,iters_start,training_time=pickle.load(open(self.model_path+"/lstm-model.info.pkl"))
                #print i_start,step_start
                
                
                
            writer = tf.train.SummaryWriter("/tmp/lstm_log", session.graph_def)
            
            # start the training loop
            for i in range(i_start,self.max_epochs):
                print "epoch "+str(i)
                epoch_size = ((len(self.data) // self.batch_size) - 1) // self.num_steps
                print "Epoch size: ",epoch_size
                session.run(tf.assign(_lr,self.learning_rate*np.power(0.75,i)))
                print "Learning Rate:",self.learning_rate*np.power(0.75,i)
                start_time = time.time()
                costs = 0.0
                iters = 0
                state = self._initial_state.eval()
                for step, (x, y) in enumerate(reader.ptb_iterator(self.data, self.batch_size,
                                                                self.num_steps)):
                    if step<step_start or (step==step_start and step>0):
                        if step==0:
                            print "fast-forwarding...",
                            costs=costs_start
                            iters=iters_start
                            start_time-=training_time
                            iters_start=0
                        if step%100==0:
                            print ".",
                        continue

                    cost, state, _ = session.run([_cost, self._final_state, _train_op],
                                                 {self._input_data: x,
                                                  self._targets: y,
                                                  self._initial_state: state})
                    costs += cost
                    iters += self.num_steps

                    if step % (epoch_size // 1000) == 10:
                        print ("%.3f perplexity: %.3f speed: %.0f wps"%
                        (step * 1.0 / epoch_size, np.exp(costs / iters),
                             iters * self.batch_size / (time.time() - start_time)))
                    # save the model and meta infos from time to time
                    if step % (epoch_size // 200) == 10:
                        tmp_start=time.time()
                        save_path = self.saver.save(session, self.model_path+"/lstm-model.ckpt")
                        
                        f=open(self.model_path+"/lstm-model.info.pkl","w")
                        pickle.dump([i,step,costs,iters,time.time()-tmp_start],f)
                        f.close()
                


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='DBLP lstm class')
    parser.add_argument('-f','--filename_embedding',required = True,
            help="Path and filename for word2vec embedding file (ending: .emb.pkl)")
    parser.add_argument('-o','--output_path',required = True,
            help='Path for output file(s)')
    parser.add_argument('-s','--sample_model',default = False,
            help="True when sampling [default: false] ")
    parser.add_argument('-n','--num_sampling_steps',default=100,type=int,
            help="Number of sampling steps ")
    parser.add_argument('-l','--num_steps',default=50,type=int,
            help="sequence length ")
    parser.add_argument('-b','--batch_size',default=24,type=int,
            help="batch_size ")
    parser.add_argument('-p','--prime_text',default=None,
            help="Text to prime RNN with when sampling")
    parser.add_argument('-c','--reload_heckpoint',default=False,
            help="When training, reload checkpoint file (if available)")
    args = parser.parse_args()
    
    # paths and filenames
    model_path=args.output_path
    filename=args.filename_embedding
    sample_model=args.sample_model
    num_sampling_steps=args.num_sampling_steps
    prime_text=args.prime_text
    
    # load embeddings
    data, count, dictionary, reverse_dictionary, saved_embedding=load_embeddings(filename)
    
    # instantiate LSTM
    myLSTM=LSTM(count, dictionary, reverse_dictionary, saved_embedding,model_path,data=data,sample_model=sample_model,num_sampling_steps=num_sampling_steps,num_steps=args.num_steps,batch_size=args.batch_size)
    
    # train or sample
    if sample_model:
        output_words,output_probs,prime_words=myLSTM.sample_model(prime_text)
    else:
        myLSTM.train_model(args.reload_heckpoint)
