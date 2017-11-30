import marshal
import jieba
import collections

stop_word = set()
for line in open("./train-data/stopwords.txt", encoding="utf-8"):
    stop_word.add(line[:-1].strip())


def read_dict(file):
    file_lines = 0
    counter = collections.Counter()
    for line in open(file, encoding="utf-8"):
        file_lines = file_lines + 1
        for w in jieba.cut(line[:-1]):
            if w in stop_word:
                continue

            counter[w] = counter[w] + 1

    return counter, file_lines


# 字典， 词的正向的概率
pos, pos_size = read_dict("./train-data/pos.txt")

# 字典， 词的负向的概率
neg, neg_size = read_dict("./train-data/neg.txt")

word_pos_prob = dict()
for w in pos:
    if w in neg:
        word_pos_prob[w] = pos[w] / (pos[w] + neg[w])
    else:
        word_pos_prob[w] = (pos[w] + 1) / (pos[w] + 2)


for w in neg:
    if w in word_pos_prob:
        continue

    word_pos_prob[w] = 1 / (neg[w] + 2)

sent_pos_prob = pos_size / (pos_size + neg_size)
file = open("./cache/model.mar", "wb")
marshal.dump([stop_word, sent_pos_prob, word_pos_prob], file, 1)
file.close()