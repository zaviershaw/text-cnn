#encoding:utf-8
import  tensorflow as tf

class TextConfig():

    embedding_size=100     #dimension of word embedding
    vocab_size=8000        #number of vocabulary
    pre_trianing = None   #use vector_char trained by word2vec

    seq_length=200         #max length of sentence
    num_classes=2          #number of labels

    num_filters=128        #number of convolution kernel
    filter_sizes=[2,3,4]   #size of convolution kernel


    hidden_dim = 128       #全连接层神经元
    keep_prob=0.5          #droppout
    lr= 1e-3                #learning rate
    decay_steps = 64      #decay iterations steps
    lr_decay= 0.8          #learning rate decay
    clip= 6.0              #gradient clipping threshold
    l2_reg_lambda=0.01     #l2 regularization lambda

    num_epochs=10          #epochs
    batch_size=64          #batch_size
    print_per_batch =50   #print result
    save_per_batch = 10    #save result to tensorboard

    train_filename='./data/email_train.txt'  #train data
    test_filename='./data/email_test.txt'    #test data
    val_filename='./data/email_val.txt'      #validation data
    vocab_filename='./data/vocab.txt'        #vocabulary
    vector_word_filename='./data/vector_word.txt'  #vector_word trained by word2vec
    vector_word_npz='./data/vector_word.npz'   # save vector_word to numpy file

class TextCNN(object):

    def __init__(self,config):

        self.config=config

        self.input_x=tf.placeholder(tf.int32,shape=[None,self.config.seq_length],name='input_x')
        self.input_y=tf.placeholder(tf.float32,shape=[None,self.config.num_classes],name='input_y')
        self.keep_prob=tf.placeholder(tf.float32,name='dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        #self.l2_loss = tf.constant(0.0)

        self.cnn()
    def cnn(self):
        # 词向量映射
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size])
            self.embedding_inputs= tf.nn.embedding_lookup(self.embedding, self.input_x)
            #self.embedding_inputs_expanded = tf.expand_dims(self.embedding_inputs, -1)

        with tf.name_scope('cnn'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                # CNN layer
                conv = tf.layers.conv1d(self.embedding_inputs, self.config.num_filters, filter_size, name='conv-%s' % filter_size)
                # global max pooling layer
                gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
                pooled_outputs.append(gmp)  

            self.h_pool = tf.concat(pooled_outputs, 1)
            #self.outputs= tf.reshape(self.h_pool, [-1, num_filters_total])


        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(self.h_pool, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        #损失函数，交叉熵
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            #self.l2_loss += tf.nn.l2_loss(fc_w)
            #self.l2_loss += tf.nn.l2_loss(fc_b)
            #self.loss = tf.reduce_mean(cross_entropy) + self.config.l2_reg_lambda * self.l2_loss
            self.loss = tf.reduce_mean(cross_entropy)

        #优化器
        with tf.name_scope('optimizer'):
            #学习率衰减
            starter_learning_rate = self.config.lr
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,self.config.decay_steps, self.config.lr_decay, staircase=True)
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            #compute_gradients()计算梯度
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            #clip_by_global_norm:修正梯度值
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            #apply_gradients()应用梯度
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        #准确率
        with tf.name_scope('accuracy'):
            correct_pred=tf.equal(tf.argmax(self.input_y,1),self.y_pred_cls)
            self.acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


