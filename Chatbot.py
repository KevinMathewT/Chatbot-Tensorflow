import tensorflow as tf
import numpy as np
import pickle
import re

question_length = 10
answer_length = 12
thought_vector_size = 512
embedding_size = 100
num_sampled = 512
learning_rate = 2e-3
batch_size = 64
steps = 5000
num_epochs_done = 25000

with open('data.pickle', 'rb') as data_file:
    tokenized_questions, tokenized_answers, question_vocab, answer_vocab, \
        question_w2id, question_id2w, answer_w2id, answer_id2w = pickle.load(data_file)

print(tokenized_questions[0])
print(tokenized_answers[0])
print(question_vocab[0])
print(answer_vocab[0])

question_vocab_size = len(question_vocab) + 2
answer_vocab_size = len(answer_vocab) + 4

print(len(tokenized_questions))
print(len(tokenized_answers))
print(question_vocab_size)
print(answer_vocab_size)
encoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None],
                                 name='encoder_{}'.format(iterator))
                  for iterator in range(question_length)]
decoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None],
                                 name='decoder_{}'.format(iterator))
                  for iterator in range(answer_length)]
decoder_targets = [decoder_inputs[iterator + 1] for iterator in range(answer_length - 1)]
decoder_targets.append(tf.placeholder(dtype=tf.int32, shape=[None], name='end_of_sentence'))
decoder_targets_weights = [tf.placeholder(dtype=tf.float32, shape=[None], name='target_weights_{}'.format(iterator))
                           for iterator in range(answer_length)]
sampled_loss_weights = tf.get_variable(name='projection_weight', shape=[answer_vocab_size, thought_vector_size],
                                       dtype=tf.float32)
bias = tf.get_variable(name='projection_bias', shape=[answer_vocab_size],
                       dtype=tf.float32)
weights = tf.transpose(sampled_loss_weights)
decoder_projection = (weights, bias)
outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
    encoder_inputs=encoder_inputs,
    decoder_inputs=decoder_inputs,
    cell=tf.contrib.rnn.BasicLSTMCell(thought_vector_size),
    num_encoder_symbols=question_vocab_size,
    num_decoder_symbols=answer_vocab_size,
    embedding_size=embedding_size,
    output_projection=decoder_projection,
    feed_previous=False,
    dtype=tf.float32
)


def sampled_smax_loss(labels, logits):
    return tf.nn.sampled_softmax_loss(
        weights=sampled_loss_weights,
        biases=bias,
        labels=tf.reshape(labels, [-1, 1]),
        inputs=logits,
        num_sampled=num_sampled,
        num_classes=answer_vocab_size,
        name='sampled_softmax_loss'
    )


loss = tf.contrib.legacy_seq2seq.sequence_loss(
    logits=outputs,
    targets=decoder_targets,
    weights=decoder_targets_weights,
    softmax_loss_function=sampled_smax_loss,
    name='chatbot_sequence_loss'
)


def padding(questions_param, answers_param):
    for iterator in range(min(len(questions_param), len(answers_param))):
        questions_param[iterator] = questions_param[iterator] \
                                    + (question_length - len(questions_param[iterator])) * [question_w2id['<PAD>']]
        answers_param[iterator] = [answer_w2id['<START>']] \
                                  + answers_param[iterator] \
                                  + [answer_w2id['<EOS>']] \
                                  + (question_length - len(answers_param[iterator])) * [answer_w2id['<PAD>']]
    return questions_param, answers_param


def feed_dict(questions_param, answers_param, batch=batch_size):
    dictionary_param = {}
    random = np.random.choice(min(len(questions_param), len(answers_param)), size=batch, replace=False)

    for iterator in range(question_length):
        dictionary_param[encoder_inputs[iterator].name] = np.array([questions_param[jiterator][iterator]
                                                                    for jiterator in random],
                                                                   dtype=np.int32)
    for iterator in range(answer_length):
        dictionary_param[decoder_inputs[iterator].name] = np.array([answers_param[jiterator][iterator]
                                                                    for jiterator in random],
                                                                   dtype=np.int32)
    dictionary_param[decoder_targets[len(decoder_targets) - 1].name] = np.full(shape=[batch],
                                                                               fill_value=answer_w2id['<PAD>'],
                                                                               dtype=np.int32)
    for iterator in range(answer_length-1):
        decoder_weights = np.ones(shape=batch, dtype=np.float32)
        temp = dictionary_param[decoder_inputs[iterator + 1].name]
        for jiterator in range(batch):
            if temp[jiterator] == answer_w2id['<PAD>']:
                decoder_weights[jiterator] = 0.0
        dictionary_param[decoder_targets_weights[iterator].name] = decoder_weights
    dictionary_param[decoder_targets_weights[answer_length - 1].name] = np.zeros(shape=batch, dtype=np.float32)

    return dictionary_param


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


def getAnswer(answer_param):
    answer_words_param = []
    for iterator in range(answer_length):
        smax = tf.nn.softmax(answer_param[iterator])
        index = np.argmax(smax)
        answer_words_param.append([index])
    return answer_words_param


outputs_projection = [tf.matmul(outputs[i], decoder_projection[0]) + decoder_projection[1]
                      for i in range(answer_length)]
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()

questions, answers = padding(tokenized_questions, tokenized_answers)

saver = tf.train.Saver()
path = tf.train.latest_checkpoint('checkpoints')

with tf.Session() as sess:
    saver.restore(sess, path)
    # sess.run(init)
    for step in range(steps + 1):
        dictionary = feed_dict(questions, answers)
        sess.run(optimizer, feed_dict=dictionary)
        loss_value = sess.run(loss, feed_dict=dictionary)
        if step % 5 == 0:
            print("Step :", step + num_epochs_done, ", Loss :", loss_value)
        if step % 500 == 0 and step != 0:
            saver.save(sess, 'checkpoints/', global_step=step + num_epochs_done)
            print("Checkpoint saved")
