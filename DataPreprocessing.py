import pickle
import re
from collections import Counter

conversations = open('datasets/movie_lines.txt', encoding='utf-8', errors='ignore').read().split(
    '\n')
conversations_ids = open('datasets/movie_conversations.txt', encoding='utf-8',
                         errors='ignore').read().split('\n')

# conversations = open('datasets\friends-final.txt', encoding='utf-8', errors='ignore').read().split('\n')

print("Examples before pre-processing : ")
print(conversations[0])

print(len(conversations))
question = []
answer = []

# for iterator in range(len(conversations)-1):
#     question_elements = conversations[iterator].split('\t')
#     answer_elements = conversations[iterator+1].split('\t')
#     # print(question_elements)
#     # print(answer_elements)
#     if len(question_elements) < 5 or len(answer_elements) < 5:
#         continue
#     if question_elements[1] == answer_elements[1]:
#         question.append(question_elements[5])
#         answer.append(answer_elements[5])


print(conversations_ids[0])

id2line = {}
for line in conversations:
    elements = line.split(' +++$+++ ')
    if len(elements) == 5:
        id2line[elements[0]] = elements[4]

conversation_lists = []
for line in conversations_ids[:-1]:
    conversation_lists.append(line.split(' +++$+++ ')[-1][2:-2].split("', '"))


for convs in conversation_lists:
    for iterator in range(len(convs) - 1):
        question.append(id2line[convs[iterator]])
        answer.append(id2line[convs[iterator + 1]])


print("Number of Question-Answer pairs before filtering : ")
print(len(question))
print(len(answer))

print("Question and Answer examples : ")
print(question[0])
print(answer[0])

print("Cleaning...")


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
    x = x.replace("'bout", " about")
    x = x.replace("'til", " until")
    x = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", x)
    x = x.replace("  ", " ")
    return x


cleaned_question = []
for item in question:
    cleaned_question.append(cleaner(item))
cleaned_answer = []
for item in answer:
    cleaned_answer.append(cleaner(item))

print("Done.")
print("After cleaning : ")
print(cleaned_question[0])
print(cleaned_answer[0])

min_filter_len = 1
max_filter_len = 10

print("Filtering out questions and answers with length less than", min_filter_len, "and greater than", max_filter_len,
      "...")


# print("Number of Question-Answer pairs after filtering : 2")
# print(len(cleaned_question))
# print(len(cleaned_answer))


filtered_questions = []
filtered_answers = []

for iterator in range(len(cleaned_question)):
    if min_filter_len <= len(cleaned_question[iterator].split()) <= max_filter_len and \
            min_filter_len <= len(cleaned_answer[iterator].split()) <= max_filter_len and \
            max(len(cleaned_question[iterator].split()), len(cleaned_answer[iterator].split())) \
            < min(len(cleaned_question[iterator].split()), len(cleaned_answer[iterator].split())) + 3:
        filtered_questions.append(cleaned_question[iterator])
        filtered_answers.append(cleaned_answer[iterator])


print("Number of Question-Answer pairs after filtering : ")
print(len(filtered_questions))
print(len(filtered_answers))

print("After filtering : ")
print(filtered_questions[0:10])
print(filtered_answers[0:10])

# vocabulary = {}
# for question in filtered_questions:
#     for word in question.split(' '):
#         if word not in vocabulary:
#             vocabulary[word] = 1
#         else:
#             vocabulary[word] += 1
# for answer in filtered_answers:
#     for word in answer.split(' '):
#         if word not in vocabulary:
#             vocabulary[word] = 1
#         else:
#             vocabulary[word] += 1

print("Creating Vocabulary and Word2Index maps and tokenizing words...")
question_vocab_dict = Counter(word for question in filtered_questions
                              for word in question.split())
answer_vocab_dict = Counter(word for answer in filtered_answers
                            for word in answer.split())

question_vocab = map(lambda x: x[0], sorted(question_vocab_dict.items(), key=lambda x: -x[1]))
answer_vocab = map(lambda x: x[0], sorted(answer_vocab_dict.items(), key=lambda x: -x[1]))

question_vocab = list(question_vocab)
answer_vocab = list(answer_vocab)
question_vocab = question_vocab[:15000]
answer_vocab = answer_vocab[:14800]


start = 2
question_w2id = dict([(word, index + start) for index, word in enumerate(list(question_vocab))])
question_w2id['<UNK>'] = 0
question_w2id['<PAD>'] = 1
question_id2w = dict([(index, word) for word, index in question_w2id.items()])
print("Popular words in the dataset: ")
print(list(question_w2id.keys())[:10])

start = 4
answer_w2id = dict([(word, index + start) for index, word in enumerate(list(answer_vocab))])
answer_w2id['<UNK>'] = 2
answer_w2id['<START>'] = 1
answer_w2id['<EOS>'] = 0
answer_w2id['<PAD>'] = 3
answer_id2w = dict([(index, word) for word, index in answer_w2id.items()])

tokenized_questions = [[question_w2id[word]
                        if word in question_w2id.keys()
                        else question_w2id['<UNK>']
                        for word in question.split()]
                       for question in filtered_questions]
tokenized_answers = [[answer_w2id[word]
                      if word in answer_w2id.keys()
                      else answer_w2id['<UNK>']
                      for word in answer.split()]
                     for answer in filtered_answers]
print("Done.")
print("Tokenizing Examples : ")
print(filtered_questions[1])
print(tokenized_questions[1])
print(filtered_answers[1])
print(tokenized_answers[1])

print("Final number of Questions-Answer pairs : ")
print(len(filtered_questions))
print(len(filtered_answers))

# tokenized_questions = []
# tokenized_answers = []
#
# for question in filtered_questions:
#     tokenized_questions.append(question.split())
#
# for answer in filtered_answers:
#     tokenized_answers.append(answer.split())

print("Saving data to data.pickle...")
with open('data.pickle', 'wb') as f:
    pickle.dump([tokenized_questions, tokenized_answers, question_vocab, answer_vocab, question_w2id, question_id2w,
                 answer_w2id, answer_id2w], f)
print("Done.")
