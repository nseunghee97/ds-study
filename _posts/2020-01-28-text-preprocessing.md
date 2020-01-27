---
title: "텍스트 전처리 (Text Preprocessing)"
last_modified_at: "2020-01-28 00:38:00 +0900"
categories:
  - NLP
tags:
  - NLP
  - Text Preprocessing
toc: true
toc_sticky: true
---

자연어 처리 기법이 잘 적용될 수 있도록, 용도에 맞게 텍스트를 사전에 처리하는 텍스트 전처리에 대해 다룬다.

## 텍스트를 토큰으로 나누기

### Tokenization (토큰화)

#### Sentence Tokenization

= Sentence Splitting, Sentence Segmentation, Sentence Detection
: **가지고 있는 말뭉치 내에서 문장 단위로 구분하는 작업**

- **Punctuation** is usually used to define sentence boundaries.
    - Not all the punctuation characters end a sentence.
    - information on the token containing the potential boundary.
    - features of the word on the left(right) of the candidate (left and right **contexts**)
        - wider contexts do not increase significantly the classifier accuracy

- **".", "?", "!"와 같은 문장 부호**가 문장의 끝에 쓰인 것인지 분류해야함. (by 기계학습)
    - 앞/뒤글자(prefix/suffix)나 토큰이 무엇인지 확인하여 **문맥상** 문장의 boundary 여부를 판단


            ~ will join the board as a nonexecutive director Nov. 29.
        - 보통 앞 뒤 두 토큰 정도만 확인


#### Word Tokenization

**: 의미 있는 최소 단위(토큰)으로 주어진 텍스트를 쪼개는 과정.**

> 한국어에서 의미를 가지는 최소 단위 : 형태소. 따라서 한국어의 경우 어절 단위가 아닌 형태소 단위 토큰화가 바람직.

토큰으로 나누는 데에는 여러 가지 기준 존재

- 공백 기준 : 공백을 기준으로 나눔
- 구두점을 별도로 분류하여 나눔 등...

원하는 결과에 따라 알맞은 tokenizer를 사용

#### POS(Part-of-Speech) Tagging

**: 나눠진 토큰에 품사를 태깅하는 것.**

한글 자연어 처리 라이브러리(KonlPy)
- [Hannanum Class](http://kldp.net/hannanum)
    - 사전/차트 기반
    - 전처리, 형태소분석, 품사태깅
- [Kkma Class](http://kkma.snu.ac.kr)
    - 확률모델 이용한 Tag 후보 선정(인접조건 검사)
    - 형태소 분석, 색인어 추출
- [Komoran Class](http://www.shineware.co.kr)
- [Mecab Class](https://bitbucket.org/eunjeon/seunjeon)
    - 사전 기반
    - 띄어 쓰기가 잘못된 오류를 교정하여 형태소 분석
    - 명사 추출, 복합명사 분해, 어절 추출
- [Twitter(OKT) Class](https://openkoreantext.org)
    - 규칙/사전기반 : 신조어 사전 업데이트
    - SNS의 특성을 잘 반영한 처리 ex. ㅋㅋㅋ

### Cleaning(정제) 및 Normalization(정규화)

**정제 : 갖고 있는 텍스트로부터 노이즈 데이터를 제거.**
**정규화 : 표현 방법이 다른 단어들을 통합시켜서 같은 단어로 만듦.**

정제 작업은 토큰화 작업에 방해가 되는 부분을 배제하기 위해 토큰화 작업보다 앞서 이루어지기도 하지만, 토큰화 작업 이후에도 여전히 남아있는 노이즈들을 제거하기위해 지속적으로 이루어짐.

> 정제 및 정규화 기법

<u>1. 규칙에 기반한 표기가 다른 단어들의 통합 (Normalization)</u>

**정규화의 목적** : 텍스트 내 서로 다른 token의 개수 줄이기. 최대한 통일.
=> 이후 분석할 때 더욱 간단하게 할 수 있기 위함.

#### Stemming (어간 추출)

**: 어형이 변형된 단어로부터 접사 등을 제거한 어간을 분리해내는 것**
어근 상 차이가 있더라도 관련된 단어가 동일한 어간으로 매핑되는 것이 목적

        "fishing","fished"."fisher">>>(stemming)>>>"fish"

#### Lemmatization (원형 복원)

**: 문법적 요인으로 인해 동일한 의미가 다른 활용형으로 사용되는 것을 복원**

=> 단/복수, 동사 시제, 소유격 등의 변형 원래의 형태로 대체

        The boy's cars are different colors >>> (lemmatization) >>> the boy car be differ color

<u>2. 대소문자 통합</u>

무작정 통합하면 안됨. ex. US(미국), us(우리)

<u>3. 불필요한 단어 제거(노이즈 데이터 제거)</u>

- 등장 빈도가 적은 단어
- 길이가 짧은 단어

##### Regular Expression (정규표현)

노이즈 데이터의 특징을 잡아냈다면, 이를 **정규표현식(Regular Expression)**[^1] 이용해 제거하는 것도 방법.

[^1]: 정규표현식 연습 사이트 1) https://regex101.com/ 2) https://regexr.com/

## 불필요한 토큰 제거

### Stopword (불용어)

: 여러 텍스트에 공통적으로 빈번히 출현하지만 주제어로서의 가치는 없는 단어

- 영어 : 대명사, 전치사, 부사 등
- 한국어 : 대명사, 의존명사, 조사, 감탄사, 수사 등

주로 token들에 대해 word counting을 실시하여, count 횟수가 높은 단어들부터 확인하여 불용어라고 판단되는 단어의 경우 불용어 사전에 추가하는 방식으로 불용어 삭제.

## 텍스트 처리 방식

### Integer Encoding (정수 인코딩)

**: 단어를 빈도수 순으로 정렬한 단어 집합(vocabulary)을 만들고, 빈도수가 높은 순서대로 차례로 낮은 숫자부터 정수를 부여하는 방법**

텍스트에 대해 word counting 실행 후 빈도수 높은 단어부터 정수 인덱스 차례대로 부여

### One-hot Encoding (원-핫 인코딩)

**: 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식**

단어 집합 : 텍스트의 모든 단어를 중복을 허용하지 않고 모아놓은 것

> 원-핫 인코딩의 과정

1. 각 단어에 고유한 인덱스를 부여. (정수 인코딩)
2. 표현하고 싶은 단어의 인덱스의 위치에 1을 부여하고, 다른 단어의 인덱스의 위치에는 0을 부여. => 원 핫 벡터


> 원-핫 인코딩의 단점

- 단어의 개수가 늘어날 수록 벡터를 저장하기 위해 필요한 공간이 계속 늘어남. (단어 집합의 크기가 곧 벡터 차원의 수)
- 단어 간 유사성 표현하지 못함

### Byte Pair Encoding (BPE, 단어 분리하기)

**: 훈련 단계에서 학습하지 못한 단어에 대해, 단어를 분리해 의미 있는 단위로 나누어 해당 단어를 이해하는 전처리 과정.**

- 하나의 단어는 의미있는 여러 내부 단어들(subwords)의 조합으로 구성된 경우가 많음.
- 따라서 단어를 여러 단어로 분리해서 단어를 이해
- 토크나이저를 통해 실행 (단어 분리 토크나이저)

=> 배운 적 없는 단어에 대해 대처 가능

### Splitting Data (데이터의 분리)

머신러닝에 있어 data를 train 데이터와 test 데이터로 나누어 머신러닝 모델에 데이터 지도 학습.

---
참고문헌

- [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155) (https://wikidocs.net/book/2155)
