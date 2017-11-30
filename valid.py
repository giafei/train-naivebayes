import jieba
import marshal
from functools import reduce

file = open("./cache/model.mar", "rb")
stop_word, sent_pos_prob, word_pos_prob = marshal.load(file)
file.close()

send_neg_prob = 1 - sent_pos_prob


def bayes(sen):
    words = []
    for w in jieba.cut(sen):
        if w in stop_word:
            continue

        if w not in word_pos_prob:
            continue

        # 正向
        t1 = sent_pos_prob * word_pos_prob[w]
        t2 = send_neg_prob * (1 - word_pos_prob[w])
        v1 = t1 / (t1 + t2)

        words.append((w, v1))

    t1 = reduce(lambda x, y: x * y, map(lambda v: v[1], words))
    t2 = reduce(lambda x, y: x * y, map(lambda v: 1 - v[1], words))

    return words, (t1 / (t1 + t2))


ss = [
    "关注这款电视很久了，等到618终于抢到了！物流真的挺快的，18号下单，19号中午就到了！这款电视太惊艳了，比我想像当中要好！",
    "首先，电视是被拆过的。我发誓我打开以后就有灰尘，半个脚印。问客服他们说是避免不了的。质量没有问题是不给调换的。最最最受不了的是电视开机有广告，看视频也有七八十秒的广告。看视频必须要买会员，不然只能看普清。",
    "客服服务态度太差，一会找不到人，一会答非所问。送货人员没有通电试机，联系客服就推给小米售后，或者说安排售后维修收费，不给解决问题。第一次在京东买东西，印象太差了",
    "满分。毕竟我已经很久没有看个电影哭得像个小孩子了 似乎大家又要说起那句老话：电影是绝佳的造梦机器。那些回不去的家乡 没能守护的家人也只有在梦里相见。",
    "人有三次死亡。第一次是生物学的死亡；第二次是社会宣布你死亡；第三次是最后一个记得你的人离开这个世界。 想念上个月去世的外婆，想念老爸，想念所有离我而去的人们。 后悔没带纸，平复了半天情绪最后离开放映厅",
    "今天放了一下午下边框跟联想电脑一样烫",
    "听说吴导身体不好又欠钱，拍出这种乱七八糟的东西就嘴下留情吧，莫名其妙的动机，莫名其妙的人物，莫名其妙的台词和莫名其妙的镜头，一颗星给最后大乱斗带来的迷之笑点。这电影公映能有六分以上我就吃键盘",
    "一群日本人和中国人，非得各种尬聊英语，台词让人不忍直视，剧情太太烂，两个大叔还要卖基，吴导江郎才尽了吗",
    "扁平庸常，毫无亮点。以及字幕翻译奇烂无比"
]

for s in ss:
    words, p = bayes(s)
    print(s)
    print(p)
    print(words)
    print("\n\n")
