---
title: "워드 임베딩(Word Embedding)"
last_modified_at: "2020-02-28 01:31:00 +0900"
categories:
  - NLP
tags:
  - NLP
  - Word Embedding
toc: true
toc_sticky: true
---

단어를 표현하는 방법 중, 단어의 의미를 벡터화시킬 수 있는 워드투벡터(Word2Vec)에 대해 다룬다.

# 워드 임베딩(Word Embedding)

[원 핫 인코딩](https://shd04121.github.io/ds-study/nlp/text-preprocessing/#one-hot-encoding-%EC%9B%90-%ED%95%AB-%EC%9D%B8%EC%BD%94%EB%94%A9)을 통한 word vector는 대부분의 행렬 값이 0으로 표현되는 **희소 벡터**(sparse vector)이다.

희소 벡터는 단어의 개수가 늘어남에 따라 벡터의 차원이 비례해서 한없이 커지기 때문에 **공간적 낭비**를 불러일으키며, 카운팅 기반이기 때문에 **단어의 의미**를 담지 못한다.  
이는 DTM도 마찬가지이다.

**워드 임베딩**은 단어를 **밀집 벡터**(dense vector)의 형태로 표현하는 방법이다.

**밀집 벡터**는 벡터의 차원을 단어 집합의 크기가 아닌, **사용자가 설정한 값**으로 모든 단어의 벡터 표현의 차원을 맞춘다.
또한 이 과정에서, 0과 1의 binary 값이 아닌 실수값을 가지게 된다.

||원-핫 인코딩|워드 임베딩|
|:-----:|:----------|:----------|
|차원|단어 집합의 크기(고차원)|사용자 지정(저차원)|
|구분|희소 벡터|밀집 벡터|
|표현 방법|수동|훈련 데이터로부터 학습|
|값의 타입|1과 0|실수|

# 워드투벡터(Word2Vec)

**워드투벡터**(Word2Vec)의 단어 벡터는 단어 간 유사도를 반영한 값을 가진다.

## 분산 표현(Distributed Representation)

'**비슷한 위치에서 등장하는 단어들은 비슷한 의미를 가진다**'는 **분포 가설**(distributional hypothesis)에 입각하여, 단어의 의미를 여러 차원에 분산하여 표현한다(다차원 공간에 벡터화).

→ **단어 간 유사도** 계산 가능

유사도 계산을 위한 학습 방법 속도를 대폭 개선시킨 Word2Vec가 많이 쓰임.

Word2Vec의 두 가지 방식

- **CBOW** : 주변에 있는 단어들을 가지고 중간에 있는 단어들을 예측하는 방법
- **Skip-Gram** : 중간에 있는 단어로 주변 단어들을 예측하는 방법

## CBOW(Continuous Bag of Words)



## Skip-Gram

## Words2Vec English

### Word2Vec 훈련

훈련 데이터 링크 : https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip

훈련 데이터는 위 링크의 데이터를 전처리한 데이터를 바탕으로 진행.  
전처리 데이터를 `result` 변수에 할당

```python
from gensim.models import word2vec
```

```python
# 모델 학습 진행
model = word2vec.Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)
```

**Word2Vec hyperparameters**

- `sentences` : 데이터
- `size` : 임베딩 된 벡터의 차원.
- `window` : 컨텍스트 윈도우 크기. 고려할 주변 단어 개수
- `min_count` : 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
- `workers` : 학습을 위한 프로세스 수. 병렬 처리 스레드 수
- `sg` = 0은 CBOW, 1은 Skip-gram.

sliding window(CBOW)

![](https://wikidocs.net/images/page/22660/%EB%8B%A8%EC%96%B4.PNG)

**Attributes of Word2Vec**

- `model.wv.most_similar` : 특정 단어와의 유사성이 가장 높은 단어들 추출
- `model.wv.doesnt_match` : 유사도 없는 단어 추출
- `model.wv.similarity` : 두 단어의 유사도 계산


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



### Word2Vec의 단어벡터


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


### Word2Vec 모델 저장 및 로드


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

## Word2Vec Korean

한국어의 경우에도 영어와 같이 진행하면 된다.   
token 선택 시 주로 명사인 token만 추출해서 사용하는 듯 하다.

한국어의 경우 전처리가 너무 오래 걸리는 관계로, 이미 만들어진 모델을 통해 간단하게 예시만 보여주고 넘어가겠다.

- [박규병님이 공개한 pre trained word2vec model](https://drive.google.com/file/d/0B0ZXk88koS2KbDhXdWg1Q2RydlU/view)

단,   
한국어가 교착어라는 특성을 감안하여, Word2vec의 문제점을 해결하기 위해 CNN을 기반으로 한 char-word 임베딩을 한국어에 적용한 [kor2vec](https://github.com/naver/kor2vec)이라는 한국어 Embedding 패키지도 존재한다.


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
model_kor.wv.doesnt_match('왕 여왕 공주 시녀'.split())
```




    '시녀'




```python
model_kor.wv.doesnt_match('프랑스 뮌헨 독일 베를린'.split())
```




    '프랑스'



# 글로브(GloVe)

**카운트 기반**(코퍼스의 전체적인 통계정보 반영 ex.LSA)
+
**예측 기반**(단어 의미의 유추 작업 ex.Word2Vec)
두 방식 모두를 사용하는 단어 임베딩 방법론

## 윈도우 기반 동시 등장 행렬(Window vased Co-occurrence Matrix)

행과 열을 **전체 단어 집합의 단어들로 구성**하고, <u>i 단어의 윈도우 크기(Window Size) 내에서 k 단어가 등장한 횟수를 i행 k열에 기재한 행렬</u>

## 동시 등장 확률(Co-occurrence Probability)

동시 등장 행렬로부터 특정 단어 i의 전체 등장 횟수를 카운트하고, 특정 단어 i가 등장했을 때 어떤 단어 k가 등장한 횟수를 카운트하여 계산한 조건부 확률

- i : 중심 단어(Center Word)
- k : 주변 단어(Context Word)

## 손실 함수(Loss function)

GloVe 모델은 동시 등장 행렬과, 동시 등장 확률을 기반으로 손실 함수를 설계한다.

## GloVe Word Embedding

[GloVe module explanation](https://github.com/maciejkula/glove-python/blob/master/glove/glove.py)

```python
from glove import Corpus, Glove
```

**Glove Parameters**

- Corpus()
    - corpus.fit
        - window : 동시 등장 행렬에 반영할 주변 단어 개수
- Glove(no_components, learning_rate)
    - no_components : number of latent dimensions
    - learning_rate : learning rate for SGD estimation
<br>
    - glove.fit
      - epochs/no_threads : number of training epochs/threads
      -  verbose : print progress messages if True


```python
corpus = Corpus()
corpus.fit(result, window=5)
# 훈련 데이터로부터 GloVe에서 사용할 동시 등장 행렬 corpus 생성

glove = Glove(no_components=100, learning_rate=0.05)

glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
# 학습에 이용할 thread와 epoch의 개수 설정.
```

`glove.most_similar(word, number)`는 입력 단어 가장 유사한 단어들의 리스트를 number개 반환한다.

```python
glove.most_similar('man')
```

    [('woman', 0.9621753707315267),
     ('guy', 0.8860281455579162),
     ('girl', 0.8609057388487154),
     ('kid', 0.8383640509911114)]

# 엘모(ELMo, Embeddings from Language Model)

[전이 학습 기반 NLP (1): ELMo](https://brunch.co.kr/@learning/12)를 참고하면  언어 모델과 ELMo에 대한 기본적인 이해를 조금 더 도울 수 있다.

ELMo는

- **사전 훈련된 언어 모델**(Pre-trained language model)을 사용해 어휘 임베딩을 생성한다. 기학습된(pre-trained) 언어 모델의 **지식을 전이**(knowledge transfer)한다.
- **문맥을 반영한 워드 임베딩**(Contextualized Word Embedding) : 주변 단어의 맥락에 따라서 어휘 임베딩이 달라진다. 이는 언어 모델을 활용하기 때문에 가능하다.
- **biLM**(Bidirectional Language Model) : 주어진 문장에서 시작부터 n개의 단어를 보고, n+1번째 단어를 맞추는 방향인 **순방향 RNN** 뿐만 아니라, 주어진 문장의 끝부터 n개의 단어를 역순으로 보고, n-1번째 단어를 맞추는 방향으로 문장을 스캔하는 **역방향 RNN** 또한 활용한다.

## ELMo의 단어 임베딩 과정

![elmo](https://wikidocs.net/images/page/33930/playwordvector.PNG)

각 층을 지난 후의 출력값을 모두 사용한다.  
(각 층의 출력값이 가진 정보는 전부 서로 다른 종류의 정보를 가지고 있으므로, 이들을 모두 활용한다는 것이 ELMo의 기본 아이디어)

1. 각 층의 출력값을 연결한다.

![elmo-concatenate](https://wikidocs.net/images/page/33930/concatenate.PNG)

2. 각 층의 출력값 별로 가중치를 준다.

![elmo-weight](https://wikidocs.net/images/page/33930/weight.PNG)

3. 각 층의 출력값을 모두 더한다.

![elmo-seightedsum](https://wikidocs.net/images/page/33930/weightedsum.PNG)

4. 벡터의 크기를 결정하는 스칼라 매개변수를 곱한다.

![elmo-scalarparameter](https://wikidocs.net/images/page/33930/scalarparameter.PNG)

(가중치와 매개변수는 훈련 과정에서 학습한다.)

▶ ELMo 표현 완성!!

ELMo 표현은 Word2vec이나 GloVe같은 기존의 임베딩 벡터와 함께 사용 가능하다.
이 때, ELMo 표현을 임베딩 벡터와 연결해서 입력으로 사용한다.

## ELMo 분류 task 예제

packages and install required


```python
!pip install tensorflow-hub
```

```python
import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K

sess = tf.Session()
K.set_session(sess)
# 세션 초기화 (tensorflow)

elmo = hub.Module("https://tfhub.dev/google/elmo/1",trainable=True)
# 텐서플로우 허브에서 ELMo 다운로드

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
```


```python
# 데이터셋 불러오기
import pandas as pd
data = pd.read_csv('/content/drive/My Drive/dslab/text-processing/spam.csv', encoding='latin-1')
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1</th>
      <th>v2</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5567</th>
      <td>spam</td>
      <td>This is the 2nd time we have tried 2 contact u...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5568</th>
      <td>ham</td>
      <td>Will Ì_ b going to esplanade fr home?</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5569</th>
      <td>ham</td>
      <td>Pity, * was in mood for that. So...any other s...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5570</th>
      <td>ham</td>
      <td>The guy did some bitching but I acted like i'd...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5571</th>
      <td>ham</td>
      <td>Rofl. Its true to its name</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5572 rows × 5 columns</p>
</div>




```python
# 데이터셋 처리
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
y_data = list(data['v1'])
X_data = list(data['v2'])
# 훈련,테스트 데이터 split
n_of_train = int(len(X_data) * 0.8)
n_of_test = int(len(X_data) - n_of_train)
import numpy as np
X_train = np.asarray(X_data[:n_of_train]) #X_data 데이터 중에서 앞의 4457개의 데이터만 저장
y_train = np.asarray(y_data[:n_of_train]) #y_data 데이터 중에서 앞의 4457개의 데이터만 저장
X_test = np.asarray(X_data[n_of_train:]) #X_data 데이터 중에서 뒤의 1115개의 데이터만 저장
y_test = np.asarray(y_data[n_of_train:]) #y_data 데이터 중에서 뒤의 1115개의 데이터만 저장
```

### ELMo와 설계 모델 연결

ELMo는 텐서플로우 허브로부터 가져온 것이기 때문에 케라스에서 사용하기 위해서는 케라스에서 사용할 수 있도록 변환해주어야함.


```python
def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), as_dict=True, signature="default")["default"]
# 데이터의 이동이 케라스 → 텐서플로우 → 케라스가 되도록 하는 함수
```

### 모델 설계


```python
from keras.models import Model
from keras.layers import Dense, Lambda, Input

input_text = Input(shape=(1,), dtype=tf.string)
embedding_layer = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text) # ELMo를 사용한 임베딩 층 거쳐
hidden_layer = Dense(256, activation='relu')(embedding_layer) # 256개 뉴런이 있는 은닉층 거친 후
output_layer = Dense(1, activation='sigmoid')(hidden_layer) # 마지막 1개의 뉴런을 통해 이진 분류 수행
model = Model(inputs=[input_text], outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```


```python
history = model.fit(X_train, y_train, epochs=1, batch_size=60)
```


```python
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
```


    테스트 정확도: 0.9803


---

본 포스팅은 개인적인 공부를 위해 아래의 문헌을 정리한 것을 기반으로 일부 내용을 추가한 것임을 밝힙니다.

참고문헌

- [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155) (https://wikidocs.net/book/2155)

- [전이 학습 기반 NLP (1): ELMo](https://brunch.co.kr/@learning/12)
