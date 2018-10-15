# -*- coding: utf-8 -*-
#prediction using model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.predict
import sys
sys.path.append("/Users/liyangyang/PycharmProjects/mypy/venv/dwb/testcnn")
import data_helpers

import tensorflow as tf
import numpy as np

sys.path.append("/Users/liyangyang/PycharmProjects/mypy/venv/dwb/github_model/a03_TextRNN")
from p8_TextRNN_model import TextRNN
import os
import pickle
from tensorflow.contrib import learn
import codecs

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",19,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 12000, "how many steps before decay learning rate.") #批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir", "/Users/liyangyang/PycharmProjects/mypy/venv/dwb/github_model/a03_TextRNN/text_rnn_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",2000,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",100,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_string("data_file", "test_set.csv", "Data source for the positive data.")
tf.app.flags.DEFINE_string("predict_target_file","/Users/liyangyang/PycharmProjects/mypy/venv/dwb/github_model/a03_TextRNN/result_rnn.csv","target file path for final prediction")
#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    # 1.load data with vocabulary of words and labels
    vocab_processor_path = '/Users/liyangyang/PycharmProjects/mypy/venv/dwb/testcnn/vocab'
    # print("end padding & transform to one hot...")
    x_train, y = data_helpers.load_data_and_labels(FLAGS.data_file)
    print('y.shape',y.shape)

    # vocab_processor = learn.preprocessing.VocabularyProcessor(2000,min_frequency=2)
    # x = np.array(list(vocab_processor.fit_transform(x_train)))
    # vocab_processor.save(vocab_processor_path)

    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_processor_path)
    testX2 = np.array(list(vocab_processor.transform(x_train)))
    vocab_size = len(vocab_processor.vocabulary_)
    print("end padding...")
   # 3.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # 4.Instantiate Model
        textRNN=TextRNN(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sequence_length,
                        vocab_size, FLAGS.embed_size, FLAGS.is_training)
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint for TextRNN")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop")
            return
        # 5.feed data, to get logits
        number_of_training_data=len(testX2)
        print("number_of_training_data:",number_of_training_data)
        index=0
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
        #for start, end in zip(range(0, number_of_training_data, FLAGS.batch_size),range(FLAGS.batch_size, number_of_training_data+1, FLAGS.batch_size)):
        for start, end in zip(range(0, number_of_training_data, FLAGS.batch_size),range(FLAGS.batch_size, number_of_training_data+1, FLAGS.batch_size)):
            logits=sess.run(textRNN.logits,feed_dict={textRNN.input_x:testX2[start:end],textRNN.dropout_keep_prob:1}) #'shape of logits:', ( 1, 1999)
            # 6. get lable using logtis
            #predicted_labels=get_label_using_logits(logits[0],vocabulary_index2word_label) #logits[0]
            # 7. write question id and labels to file system.
            #write_question_id_with_labels(question_id_list[index],predicted_labels,predict_target_file_f)
            #############################################################################################################
            print("start:",start,";end:",end)
            question_id_sublist=y[start:end]
            get_label_using_logits_batch(question_id_sublist, logits, predict_target_file_f)
            ########################################################################################################
            index=index+1
        predict_target_file_f.close()

# get label using logits
def get_label_using_logits_batch(question_id_sublist,logits_batch,f):
    print("get_label_using_logits.shape:", logits_batch.shape) # (10, 1999))=[batch_size,num_labels]===>需要(10,5)
    for i,logits in enumerate(logits_batch):
        lable = int(np.argmax(logits))+1  # print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
        # print(question_id_sublist[i],lable)
        write_question_id_with_labels(question_id_sublist[i], lable, f)
    f.flush()
# write question id and labels to file system.
def write_question_id_with_labels(question_id,lable,f):
    f.write(str(question_id)+","+str(lable)+"\n")

if __name__ == "__main__":
    tf.app.run()