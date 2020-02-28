

```python
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
```

following packages are needed.

- gensim

# Word2Vec English


```python
# 미리 전처리해둔 array 파일 불러오기
result = np.load('result.npy')
result = result.tolist() # np array를 list 형식으로 변환
```


```python
result[:3]
```




    [['here',
      'are',
      'two',
      'reasons',
      'companies',
      'fail',
      'they',
      'only',
      'do',
      'more',
      'of',
      'the',
      'same',
      'or',
      'they',
      'only',
      'do',
      'what',
      's',
      'new'],
     ['to',
      'me',
      'the',
      'real',
      'real',
      'solution',
      'to',
      'quality',
      'growth',
      'is',
      'figuring',
      'out',
      'the',
      'balance',
      'between',
      'two',
      'activities',
      'exploration',
      'and',
      'exploitation'],
     ['both',
      'are',
      'necessary',
      'but',
      'it',
      'can',
      'be',
      'too',
      'much',
      'of',
      'a',
      'good',
      'thing']]



## Word2Vec 훈련


```python
from gensim.models import word2vec
```


```python
# 모델 학습 진행
model = word2vec.Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)
```

Word2Vec hyperparameters

- sentences : 데이터
- size : 임베딩 된 벡터의 차원.
- window : 컨텍스트 윈도우 크기. 고려할 주변 단어 개수
- min_count : 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
- workers : 학습을 위한 프로세스 수. 병렬 처리 스레드 수
- sg = 0은 CBOW, 1은 Skip-gram.

sliding window

![](https://wikidocs.net/images/page/22660/%EB%8B%A8%EC%96%B4.PNG)

Attributes of Word2Vec

- model.wv.most_similar : 특정 단어와의 유사성이 가장 높은 단어들 추출
- model.wv.doesnt_match : 유사도 없는 단어 추출
- model.wv.similarity : 두 단어의 유사도 계산


```python
model.wv.most_similar('man')
```




    [('woman', 0.8435827493667603),
     ('guy', 0.7959291338920593),
     ('lady', 0.7813702821731567),
     ('boy', 0.7739896774291992),
     ('girl', 0.755851686000824),
     ('gentleman', 0.749406099319458),
     ('soldier', 0.7424408793449402),
     ('kid', 0.7139174938201904),
     ('surgeon', 0.685417890548706),
     ('doctor', 0.6692849397659302)]




```python
model.wv.most_similar('film')
```




    [('movie', 0.7977195978164673),
     ('book', 0.7495905756950378),
     ('painting', 0.7327531576156616),
     ('photo', 0.7249643802642822),
     ('photograph', 0.7244656085968018),
     ('video', 0.7005455493927002),
     ('song', 0.6778708696365356),
     ('cartoon', 0.6639515161514282),
     ('poem', 0.6605449318885803),
     ('sculpture', 0.6559155583381653)]




```python
model.wv.doesnt_match('man woman child kitchen'.split())
```




    'kitchen'




```python
model.wv.doesnt_match('france england germany berlin'.split())
```




    'berlin'




```python
model.wv.similarity('man','woman')
```




    0.843582699798206



## Word2Vec의 단어벡터


```python
# 모델의 단어 빈도수 리스트화
vocab = list(model.wv.vocab)
```


```python
# 단어마다 벡터값으로 변환
word_vector = model[vocab]
```


```python
print(word_vector)
```

    [[ 0.13817619 -0.7706773  -0.8286377  ...  0.17833184  0.8539625
      -0.9501489 ]
     [ 2.0446525   2.8240228   0.67058825 ... -1.9625183  -1.0110266
       1.9254017 ]
     [ 0.9581023  -1.1653852   0.20218755 ... -0.30519286  0.8256667
      -1.3964056 ]
     ...
     [ 0.00662448 -0.0505872  -0.02543428 ...  0.00455616 -0.09347355
      -0.04729037]
     [-0.0486829  -0.06505231 -0.03188318 ...  0.01894769  0.03513615
      -0.01040651]
     [-0.05353064 -0.05926805 -0.02057108 ... -0.02632536 -0.00692036
      -0.02110729]]
    

## Word2Vec 모델 저장 및 로드


```python
from gensim.models import KeyedVectors
```


```python
model.wv.save_word2vec_format('word2vec_model') # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format('word2vec_model') # 모델 로드
```


```python
loaded_model.wv.most_similar('man') # 원래의 모델과 같은 값이 나옴
```




    [('woman', 0.8435827493667603),
     ('guy', 0.7959291338920593),
     ('lady', 0.7813702821731567),
     ('boy', 0.7739896774291992),
     ('girl', 0.755851686000824),
     ('gentleman', 0.749406099319458),
     ('soldier', 0.7424408793449402),
     ('kid', 0.7139174938201904),
     ('surgeon', 0.685417890548706),
     ('doctor', 0.6692849397659302)]



혹은 직접 훈련한 모델이 아닌 사전 훈련된 Word2Vec 임베딩 모델 사용 가능

- [구글 임베딩 모델](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
- [위키피디아 데이터](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2)

위의 모델 다운로드 후,
`gensim.models.KeyedVectors.load_word2vec_format` 함수를 이용하여 모델 로드 후 사용


```python

```


```python

```

# Word2Vec Korean

한국어의 경우에도 영어와 같이 진행하면 된다.  
token 선택 시 주로 명사인 token만 추출해서 사용하는 듯 하다.

한국어의 경우 전처리가 너무 오래 걸리는 관계로, 이미 만들어진 모델을 통해 간단하게 예시만 보여주고 넘어가겠다.

- [박규병님이 공개한 pre trained word2vec model](https://drive.google.com/file/d/0B0ZXk88koS2KbDhXdWg1Q2RydlU/view)


```python
import gensim
model_kor = gensim.models.Word2Vec.load('ko.bin')
```


```python
model_kor.wv.most_similar('강아지')
```




    [('고양이', 0.7290453314781189),
     ('거위', 0.7185634970664978),
     ('토끼', 0.7056223750114441),
     ('멧돼지', 0.6950401067733765),
     ('엄마', 0.693433403968811),
     ('난쟁이', 0.6806551218032837),
     ('한마리', 0.6770296096801758),
     ('아가씨', 0.675035297870636),
     ('아빠', 0.6729634404182434),
     ('목걸이', 0.6512461304664612)]




```python
model_kor.wv.doesnt_match('영국 프랑스 독일 베를린'.split())
```




    '베를린'




```python
model_kor.wv.doesnt_match('프랑스 뮌헨 독일 베를린'.split())
```




    '프랑스'




```python

```


```python

```
