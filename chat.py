import pickle
import re
import numpy as np
import tensorflow as tf

with open('data.pickle', 'rb') as data_file:
    tokenized_questions, tokenized_answers, question_vocab, answer_vocab, \
        question_w2id, question_id2w, answer_w2id, answer_id2w = pickle.load(data_file)

question_length = 10
answer_length = 12
thought_vector_size = 512
embedding_size = 100
num_sampled = 512
learning_rate = 2e-3
batch_size = 64
epochs = 1000
question_vocab_size = len(question_vocab) + 2
answer_vocab_size = len(answer_vocab) + 4


def cleaner(x):
    x = x.lower()
    x = x.replace("aren't", "are not")
    x = x.replace("can't", "cannot")
    x = x.replace("couldn't", "could not")
    x = x.replace("didn't", "did not")
    x = x.replace("doesn't", "does not")
    x = x.replace("don't", "do not")
    x = x.replace("hadn't", "had not")
    x = x.replace("hasn't", "has not")
    x = x.replace("haven't", "have not")
    x = x.replace("he'd", "he had")
    x = x.replace("he'll", "he will")
    x = x.replace("he's", "he is")
    x = x.replace("I'd", "I had")
    x = x.replace("I'll", "I will")
    x = x.replace("I'm", "I am")
    x = x.replace("I've", "I have")
    x = x.replace("isn't", "is not")
    x = x.replace("let's", "let us")
    x = x.replace("mightn't", "might not")
    x = x.replace("mustn't", "must not")
    x = x.replace("shan't", "shall not")
    x = x.replace("she'd", "she had")
    x = x.replace("she'll", "she will")
    x = x.replace("she's", "she is")
    x = x.replace("shouldn't", "should not")
    x = x.replace("that's", "that is")
    x = x.replace("there's", "there is")
    x = x.replace("they'd", "they had")
    x = x.replace("they'll", "they will")
    x = x.replace("they're", "they are")
    x = x.replace("they've", "they have")
    x = x.replace("we'd", "we had")
    x = x.replace("we're", "we are")
    x = x.replace("we've", "we have")
    x = x.replace("weren't", "were not")
    x = x.replace("what'll", "what will")
    x = x.replace("what're", "what are")
    x = x.replace("what's", "what is")
    x = x.replace("what've", "what have")
    x = x.replace("where's", "where is")
    x = x.replace("who's", "who had")
    x = x.replace("who'll", "who will")
    x = x.replace("who're", "who are")
    x = x.replace("who's", "who is")
    x = x.replace("who've", "who have")
    x = x.replace("won't", "will not")
    x = x.replace("wouldn't", "would not")
    x = x.replace("you'd", "you had")
    x = x.replace("you'll", "you will")
    x = x.replace("you're", "you are")
    x = x.replace("you've", "you have")
    x = x.replace("'d", " would")
    x = x.replace("'ll", " will")
    x = x.replace("'re", " are")
    x = x.replace("'ve", " have")
    x = x.replace("'bout", "about")
    x = x.replace("'til", "until")
    x = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", x)
    x = x.replace("  ", " ")
    return x


def softmax(x):
    n = np.max(x)
    e_x = np.exp(x - n)
    return e_x / e_x.sum()


def getAnswer(answer):
    answer_words = []
    for iterator in range(answer_length):
        smax = softmax(answer[iterator])
        index = np.argmax(smax)
        answer_words.append(answer_id2w[index])
    return answer_words


encoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None], name="encoder_{}".format(iterator))
                  for iterator in range(question_length)]
decoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None], name="decoder_{}".format(iterator))
                  for iterator in range(answer_length)]

sampled_loss_weights = tf.get_variable(name='projection_weight', shape=[answer_vocab_size, thought_vector_size],
                                       dtype=tf.float32)
bias = tf.get_variable(name='projection_bias', shape=[answer_vocab_size],
                       dtype=tf.float32)
weights = tf.transpose(sampled_loss_weights)
output_projection = (weights, bias)

outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
    encoder_inputs=encoder_inputs,
    decoder_inputs=decoder_inputs,
    cell=tf.contrib.rnn.BasicLSTMCell(thought_vector_size),
    num_encoder_symbols=question_vocab_size,
    num_decoder_symbols=answer_vocab_size,
    embedding_size=embedding_size,
    feed_previous=True,
    output_projection=output_projection,
    dtype=tf.float32
)

out_projection = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1]
                  for i in range(answer_length)]

saver = tf.train.Saver()
path = tf.train.latest_checkpoint('checkpoints')

with tf.Session() as sess:
    saver.restore(sess, path)

    while True:
        print("-------------------------------------------------------")
        question = input()
        print(question)
        question = cleaner(question)
        question = question.split()
        for iterator in range(len(question)):
            if question[iterator] not in question_w2id.keys():
                question[iterator] = question_w2id['<UNK>']
            else:
                question[iterator] = question_w2id[question[iterator]]

        print(question)
        question = question + (question_length - len(question)) * [answer_w2id['<PAD>']]

        dictionary = {}
        for iterator in range(question_length):
            dictionary[encoder_inputs[iterator].name] = np.array([question[iterator]], dtype=np.int32)

        dictionary[decoder_inputs[0].name] = np.array([answer_w2id['<START>']], dtype=np.int32)

        output = sess.run(out_projection, feed_dict=dictionary)
        output = getAnswer(output)
        final_output = ''

        for word in output:
            if word not in ['<START>', '<EOS>', '<PAD>']:
                final_output += word + " "

        print(final_output)
