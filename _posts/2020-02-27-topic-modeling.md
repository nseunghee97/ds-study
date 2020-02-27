---
title: "토픽 모델링(Topic Modeling)"
last_modified_at: "2020-02-27 23:31:00 +0900"
categories:
  - NLP
tags:
  - NLP
  - Topic Modeling
  - python
toc: true
toc_sticky: true
---

문서 집합의 추상적인 주제를 발견하여, 텍스트 본문의 숨겨진 의미 구조를 발견하기 위해 사용되는 텍스트 마이닝 기법인 토픽 모델링을 정리한다.

# 잠재 의미 분석 (LSA, Latent Semantic Analysis)

**BoW**(Bag of Words)에 기반한 DTM(Document Term Matrix)이나 TF-IDF(단어 빈도-역문헌 빈도)는 기본적으로 **단어의 빈도 수**를 이용한 <u>수치화 방식</u>으로 단어의 의미를 고려하지는 못한다는 한계가 존재한다.

따라서 DTM의 잠재된(Latent) 의미를 이끌어내기 위해 **잠재 의미 분석(LSA, Latent Semantic Analysis)** 을 사용하여 분석한다.

## 특이값 분해(SVD, Singular Value Decomposition)

특이값 분해란 A가 m x n 행렬일 때, 3개의 행렬의 곱으로 분해하는 것을 말한다.

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>A</mi>
  <mo>=</mo>
  <mi>U</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mo>&#x3A3;</mo>
  </mrow>
  <msup>
    <mi>V</mi>
    <mtext>T</mtext>
  </msup>
</math>

</br>


<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>U</mi>
  <mo>:</mo>
  <mi>m</mi>
  <mo>&#xD7;</mo>
  <mi>m</mi>
  <mtext>&#xA0;</mtext>
  <mtext>직교행렬</mtext>
  <mtext>&#xA0;</mtext>
  <mo stretchy="false">(</mo>
  <mi>A</mi>
  <msup>
    <mi>A</mi>
    <mtext>T</mtext>
  </msup>
  <mo>=</mo>
  <mi>U</mi>
  <mo stretchy="false">(</mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mo>&#x3A3;</mo>
  </mrow>
  <msup>
    <mrow class="MJX-TeXAtom-ORD">
      <mo>&#x3A3;</mo>
    </mrow>
    <mtext>T</mtext>
  </msup>
  <mo stretchy="false">)</mo>
  <msup>
    <mi>U</mi>
    <mtext>T</mtext>
  </msup>
  <mo stretchy="false">)</mo>
</math>

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>V</mi>
  <mo>:</mo>
  <mi>n</mi>
  <mo>&#xD7;</mo>
  <mi>n</mi>
  <mtext>&#xA0;</mtext>
  <mtext>직교행렬</mtext>
  <mtext>&#xA0;</mtext>
  <mo stretchy="false">(</mo>
  <msup>
    <mi>A</mi>
    <mtext>T</mtext>
  </msup>
  <mi>A</mi>
  <mo>=</mo>
  <mi>V</mi>
  <mo stretchy="false">(</mo>
  <msup>
    <mrow class="MJX-TeXAtom-ORD">
      <mo>&#x3A3;</mo>
    </mrow>
    <mtext>T</mtext>
  </msup>
  <mrow class="MJX-TeXAtom-ORD">
    <mo>&#x3A3;</mo>
  </mrow>
  <mo stretchy="false">)</mo>
  <msup>
    <mi>V</mi>
    <mtext>T</mtext>
  </msup>
  <mo stretchy="false">)</mo>
</math>
</br>

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow class="MJX-TeXAtom-ORD">
    <mo>&#x3A3;</mo>
  </mrow>
  <mo>:</mo>
  <mi>m</mi>
  <mo>&#xD7;</mo>
  <mi>n</mi>
  <mtext>&#xA0;</mtext>
  <mtext>직사각 대각행렬
  </mtext>
</math>

</br>
</br>

이 때, SVD를 통해 나온 대각행렬 Σ의 **주대각원소**를 행렬 A의 **특이값**(singular value)라고 하며, 이 특이값들은 Σ 내에서 **내림차순으로 정렬**되어있다는 특징을 가진다.

## 절단된 특이값 분해(Truncated SVD)

위에서 설명한 SVD를 full SVD라고 한다.

그러나, LSA의 경우 full SVD에서 나온 3개의 행렬에서 일부 벡터들을 삭제시켜 차원을 축소한 **절단된 SVD**(truncated SVD)를 사용한다.

![truncated-svd](https://wikidocs.net/images/page/24949/svd%EC%99%80truncatedsvd.PNG)
출처 : [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/24949)

절단된 SVD는 대각 행렬 Σ의 대각 원소의 값 중 **상위 t개만** 남게 된다. (즉, 특이값 중 **값이 큰 상위 t개만** 남게 된다)

절단된 SVD를 통해 데이터의 차원을 줄이게 되면, 계산 비용이 낮아지며, 설명력이 낮은 정보를 삭제하는 효과를 가져다주어 **기존의 행렬에서는 드러나지 않았던 층적인 의미를 확인할 수 있게 된다.**

## 잠재 의미 분석(LSA, Latent Semantic Analysis)

LSA는 기본적으로 DTM이나 TF-IDF 행렬에 절단된 SVD(truncated SVD)를 사용하여 차원을 축소시키고, 단어들의 잠재적인 의미를 끌어낸다는 아이디어를 갖고 있다.

LSA는 쉽고 빠르게 구현 가능하며 단어의 잠재적인 의미를 이끌어낼 수 있어 문서 유사도 계산 등에서 좋은 성능을 보여주지만,  
SVD의 특성상 새로운 정보에 대한 업데이트가 어렵다.

그래서 최근에는 LSA 대신 **Word2Vec** 등 단어의 의미를 벡터화할 수 있는 또 다른 방법론인 인공 신경망 기반의 방법론이 각광받고 있다.

### 실습

실습을 위해 DTM을 생성한다.

- `numpy` 패키지를 통해 직접 DTM을 생성하거나 (이후 특이값분해에서도 해당 패키지 사용하므로 import)
- `sklearn`패키지에서 `CountVectorizor`모듈을 import하여 생성한다.

  ```python
  from sklearn.feature_extraction.text import CountVectorizer
  cv = CountVectorizor(max_features=100, stop_words=stopsords)
  # max_features : DTM에 사용할 최대(max) 단어(feature) 수
  # stop_words : 분석에 필요없는 토큰 제거. 문자열('english'), 리스트 또는 None(디폴트)
  #token_pattern : 토큰 정의용 정규표현식
  # tokenizer : token 생성 함수. 함수 또는 None.
  tdm = cv.fit_transform(text)
  # text는 역토큰화된 텍스트
  tdm
  ```
  ```python
  cv.get_feature_names()
  # 사용된 feature(단어)의 리스트를 반환한다.
  ```

`CountVectorizer`는 기본적으로 토큰화가 되어있지 않은 텍스트 데이터를 입력으로 사용하기 때문에, 이를 사용하기 위해 토큰화 작업을 역으로 취소하는 역토큰화(Detokenization)작업을 수행해야 한다.(`join`함수 이용)


tf-idf 가중치를 적용한 DTM을 생성할 수도 있다. (성능이 더욱 좋아진다.)


- `sklearn`패키지에서 `TfidfVectorizer`모듈을 import하여 생성한다.

    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english', lowercase=True, max_features=1000)
    # max_features : feature로 삼을 단어의 최대 개수
    # stop_words : 분석에 필요없는 데이터라고 생각되는 부분을 제거
    tdm = tfidf.fit_transform(text)
    tdm
    ```

`TfidfVectorizor`는 기본적으로 토큰화가 되어있지 않은 텍스트 데이터를 입력으로 사용하기 때문에, 이를 사용하기 위해 토큰화 작업을 역으로 취소하는 역토큰화(Detokenization)작업을 수행해야 한다.(`join`함수 이용)

- 존재하는 DTM을 tf-idf matrix로 바꿀 경우 : `sklearn` 패키지에서 `TfidfTransformer` 모듈을 import하여 바꿔준다.

    ```python
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer()
    tfidftdm = tfidf.fit_transform(tdm)
    # tdm에 기존에 존재하는 DTM을 넣어준다.
    ```


#### Full SVD

```python
U, s, VT = np.linalg.svd(A, full_matrices = True)
```
- `U`,`VT` : np 배열로 이루어진 행렬 반환
- `s` : 특이값의 리스트 반환

#### Truncated SVD

```python
from sklearn.decomposition import TruncatedSVD
svd_model = TruncatedSVD(n_components=20, algorithm="randomized", n_iter=5, random_state=None)
# n_components : 토픽의 숫자
svd_model.fit(X)
# X : DTM
```

# 잠재 디리클레 할당(LDA, Latent Dirichlet Allocation)

LDA는 **문서의 집합으로부터 어떤 토픽이 존재하는지를 알아내기 위한 알고리즘**이다.

LDA는 **각 문서의 토픽 분포**(문서에 특정 토픽이 존재할 확률)와 **각 토픽 내의 단어 분포**(문서 내의 단어가 특정 토픽에 존재할 확률)를 추정한 후, **두 확률을 결합**하여 토픽을 추정한다.

예시)

```
문서1 : 저는 사과랑 바나나를 먹어요
문서2 : 우리는 귀여운 강아지가 좋아요
문서3 : 저의 깜찍하고 귀여운 강아지가 바나나를 먹어요
```

- 각 문서의 토픽 분포 추정
    - 문서1 : 토픽 A 100%
    - 문서2 : 토픽 B 100%
    - 문서3 : 토픽 B 60%, 토픽 A 40%

- 각 토픽의 단어 분포 추정
    - 토픽A : 사과 20%, 바나나 40%, 먹어요 40%, 귀여운 0%, 강아지 0%, 깜찍하고 0%, 좋아요 0%
    - 토픽B : 사과 0%, 바나나 0%, 먹어요 0%, 귀여운 33%, 강아지 33%, 깜찍하고 16%, 좋아요 16%

→ 문서3에서 토픽 B가 존재할 확률은 60%이고, 토픽 B에 `강아지`라는 단어가 존재할 확률은 33%이다.
토픽 A가 존재할 확률은 40%이고, 토픽 A에 `강아지`라는 단어가 존재할 확률은 0%이다.
토픽 A와 B에서의 `강아지` 등장 확률은 문서3때문에 조정된다.
각 경우를 결합하였을 때, 토픽 B의 확률이 더 높다...
그리고 이 과정 반복...
이라는 식인 것 같은데 사실 정확하게 이해하지는 못했습니당

## LDA 수행과정

**LDA의 구체적인 수행 과정**은 다음과 같다.

1. **분석자가 토픽의 개수 k를 하이퍼 파라미터로 입력한다.**
LDA는 k개의 토픽이 M개의 전체 문서에 걸쳐 분포되어 있다고 가정한다.
</br>

2. **LDA는 모든 단어를 k개 중 하나의 토픽에 할당한다.**
LDA는 모든 문서의 모든 단어에 대해 k개 중 하나의 토픽을 랜덤으로 할당한다. (랜덤이기에 결과는 전부 틀리다.)
=> 각 문서는 토픽을 가진 상태이며, 토픽은 단어 분포를 가지는 상태이다.
</br>

3. **모든 문서의 모든 단어에 대해서 아래의 사항을 반복 진행한다.** (iterative)
    1. 어떤 문서의 각 단어 w는 자신은 잘못된 토픽에 할당되어져 있지만, 다른 단어들은 전부 올바른 토픽에 할당되어져 있는 상태라고 가정한 후, 아래의 두 가지 기준에 따라 단어 w에 대해 토픽을 재할당한다.
        - p(topic t | document d) : 문서 d의 단어들 중 토픽 t에 해당하는 단어들의 비율
        - p(word w | topic t) : 단어 w를 갖고 있는 모든 문서들 중 토픽 t가 할당된 비율


LDA가 토픽의 제목을 정해주지 않지만, 실행 결과를 통해 각각의 토픽이 어떠한 단어를 담고 있는 토픽인지 판단 가능하다.

- LSA : DTM을 차원 축소 하여 축소 차원에서 근접 단어들을 토픽으로 묶는다.
- LDA : 단어가 특정 토픽에 존재할 확률과 문서에 특정 토픽이 존재할 확률을 결합확률로 추정하여 토픽을 추출한다.

```python
from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=777,max_iter=1)
# n_components = 토픽의 개수

lda_top=lda_model.fit_transform(X)
# X : DTM
```

`lda_model.components_` :  토픽 내에서의 단어 확률분포. (토픽 개수 x 단어 개수)의 배열 형태.

토픽 내의 단어 분포를 볼 수 있는 함수 정의

```python
def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])
get_topics(lda_model.components_,tfidf.get_feature_names())
# tfidf는 1000개의 단어가 저장된 단어집합. 위의 TfidfVectorizer에서 정의.
```

---

본 포스팅은 개인적인 공부를 위해 아래의 문헌을 정리한 것을 기반으로 일부 내용을 추가한 것임을 밝힙니다.

참고문헌

- [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155) (https://wikidocs.net/book/2155)
