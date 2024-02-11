from janome.tokenizer import Tokenizer
from wordcloud import WordCloud
from collections import Counter


def upload_cloud(file_path):
    token = Tokenizer()

    with open(file_path, mode='r', encoding='utf-8') as f:
        text = f.read()
        token_text = token.tokenize(text)

    words = ''
    words_list = []
    for i in token_text:
        print(f"{i.part_of_speech.split(',')}")
        pos = i.part_of_speech.split(',')[0]
        word = i.surface
        stopwords = ['こと', 'もの', 'それ', 'の', 'これ', 'ため', 'ん']
        if pos == '名詞' and word not in stopwords:
            words = words + ' ' + word
            words_list.append(word)

    wc = WordCloud(background_color="white",
                   font_path=r'C:/Windows/Fonts/HGRSGU.TTC',
                   width=800, height=800,
                   stopwords=stopwords
                   )
    wc.generate(words)
    wc.to_file('static/' + file_path[:-3] + 'png')
    result_path = file_path[:-3] + 'png'

    counter = Counter(words_list)
    top_20 = counter.most_common(20)
    print(top_20)

    return result_path

if __name__ == '__main__':
    print("start")
    result_path = upload_cloud('sample_text1.txt')
    print(result_path)    
