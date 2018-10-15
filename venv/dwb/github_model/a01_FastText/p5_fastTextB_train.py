# -*- coding: utf-8 -*-
# training the model.
# process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
import sys
sys.path.append("/Users/liyangyang/PycharmProjects/mypy/venv/dwb/testcnn")
import data_helpers

import tensorflow as tf
import numpy as np

sys.path.append("/Users/liyangyang/PycharmProjects/mypy/venv/dwb/github_model/a01_FastText")
from p5_fastTextB_model import fastTextB as fastText
# from p4_zhihu_load_data import load_data, create_voabulary, create_voabulary_label
from tflearn.data_utils import to_categorical, pad_sequences
import os
from tensorflow.contrib import learn

# import word2vec
import pickle

# configuration
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_file", "train_set.csv", "Data source for the positive data.")
tf.app.flags.DEFINE_integer("label_size", 19, "number of label")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate.")  # 批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.8, "Rate of decay for learning rate.")  # 0.5一次衰减多少
tf.app.flags.DEFINE_integer("num_sampled", 1, "The number of classes to randomly sample per batch.")  # 100
tf.app.flags.DEFINE_string("ckpt_dir", "/Users/liyangyang/PycharmProjects/mypy/venv/dwb/github_model/a01_FastText/fast_text_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len", 2000, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 128, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 100, "train size")
tf.app.flags.DEFINE_integer("validate_every", 3, "Validate every validate_every epochs.")  # 每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")
# tf.app.flags.DEFINE_string("cache_path", "/Users/liyangyang/PycharmProjects/mypy/venv/dwb/hithub_model/a01_FastText/fast_text_checkpoint/data_cache.pik", "checkpoint location for the model")


# 1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    # 1.load data(X:list of lint,y:int).
    # if os.path.exists(FLAGS.cache_path):  # 如果文件系统中存在，那么加载故事（词汇表索引化的）
    #    with open(FLAGS.cache_path, 'r') as data_f:
    #        trainX, trainY, testX, testY, vocabulary_index2word=pickle.load(data_f)
    #        vocab_size=len(vocabulary_index2word)
    # else:
    if 1 == 1:
        vocab_processor_path = '/Users/liyangyang/PycharmProjects/mypy/venv/dwb/testcnn/vocab'
        # print("end padding & transform to one hot...")
        x_train, y = data_helpers.load_data_and_labels(FLAGS.data_file)


        # vocab_processor = learn.preprocessing.VocabularyProcessor(2000,min_frequency=1)
        # x = np.array(list(vocab_processor.fit_transform(x_train)))
        # vocab_processor.save(vocab_processor_path)

        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_processor_path)
        x = np.array(list(vocab_processor.transform(x_train)))

        trainX = x[:80000]
        testX = x[80000:]
        trainY = y[:80000]
        testY = y[80000:]
        vocab_size = len(vocab_processor.vocabulary_)
        print('vocab_size',vocab_size)


    # 2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Instantiate Model
        fast_text = fastText(FLAGS.label_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                             FLAGS.decay_rate, FLAGS.num_sampled, FLAGS.sentence_len, vocab_size, FLAGS.embed_size,
                             FLAGS.is_training)
        # Initialize Save
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:  # load pre-trained word embedding
                # word_embedding = tf.constant(trainX, dtype=tf.float32)  # convert to tensor
                # t_assign_embedding = tf.assign(fast_text.Embedding,
                #                            word_embedding)  # assign this value to our embedding variables of our model
                # sess.run(t_assign_embedding)
                assign_pretrained_word_embedding(sess, trainX, vocab_size, fast_text)

        curr_epoch = sess.run(fast_text.epoch_step)
        # 3.feed data & training
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        for epoch in range(curr_epoch, FLAGS.num_epochs):  # range(start,stop,step_size)
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),
                                  range(batch_size, number_of_training_data, batch_size)):
                if epoch == 0 and counter == 0:
                    print("trainX[start:end]:", trainX[start:end])
                    print("trainY[start:end]:", trainY[start:end])
                curr_loss, curr_acc, _ = sess.run([fast_text.loss_val, fast_text.accuracy, fast_text.train_op],
                                                  feed_dict={fast_text.sentence: trainX[start:end],
                                                             fast_text.labels: trainY[start:end]})
                loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
                if counter % 10 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" % (
                        epoch, counter, loss / float(counter), acc / float(counter)))

            # epoch increment
            print("going to increment epoch counter....")
            sess.run(fast_text.epoch_increment)

            # 4.validation
            print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every == 0))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(sess, fast_text, testX, testY, batch_size)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))

                # save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=fast_text.epoch_step)  # fast_text.epoch_step

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss, test_acc = do_eval(sess, fast_text, testX, testY, batch_size)
    pass


def assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, fast_text):
    print("using pre-trained word emebedding.started...")
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
    # word2vec_model = word2vec.load('zhihu-word2vec-multilabel.bin-100', kind='bin')
    word2vec_dict = {}
    # for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
    #     word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    for i in range(1, vocab_size):  # loop each word
        word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(fast_text.Embedding,
                                   word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("using pre-trained word emebedding.ended...")


# 在验证集上做验证，报告损失、精确度
def do_eval(sess, fast_text, evalX, evalY, batch_size):
    number_examples = len(evalX)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        curr_eval_loss, curr_eval_acc, = sess.run([fast_text.loss_val, fast_text.accuracy],
                                                  feed_dict={fast_text.sentence: evalX[start:end],
                                                             fast_text.labels: evalY[start:end]})
        eval_loss, eval_acc, eval_counter = eval_loss + curr_eval_loss, eval_acc + curr_eval_acc, eval_counter + 1
    return eval_loss / float(eval_counter), eval_acc / float(eval_counter)


if __name__ == "__main__":
    tf.app.run()
