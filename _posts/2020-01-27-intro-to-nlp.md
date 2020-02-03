---
title: "Introduction to NLP"
last_modified_at: "2020-01-27 23:29:00 +0900"
categories:
  - NLP
tags:
  - NLP
toc: true
toc_sticky: true
---

자연어처리는 인공지능 분야로부터 파생된 영역으로, 인공지능은 기계가 생각하고 판단할 수 있도록 인공적으로 지능을 만드는 분야이다.  
인공적인 지능 생성을 위해서는 인간의 언어를 이해할 수 있는 기능이 요구되며, 따라서 컴퓨터가 인간의 언어인 자연어를 이해하고, 처리할 수 있도록 하는 자연어 처리가 중요하게 되었다.

NLP는 **인간의 언어가 사용되는 실세계의 모든 영역**에서 활용될 수 있으며, 그 예로는

- 정보검색, 질의응답시스템
- 기계번역, 자동통역
- 문서작성, 문서요약, 문서 분류, 철자 오류 검색 및 수정, 문법 오류 검사 및 수정 등

이 있다.

## 자연어 분석 단계

![text-preprocessing-step](https://user-images.githubusercontent.com/35002380/73628767-804f3480-4694-11ea-972d-6c617bd1aeca.jpg)


**자연어 분석 단계**는 다음과 같은 4단계를 통해 이루어진다.


### 1. 형태소 분석 (Morphological Analysis)

**: 입력된 문자열을 분석해 형태소라는 최소 의미 단위로 분리**

<br>

> 형태소 분석의 난점

- 중의성
ex. "감기는" 의 경우 세 가지 의미로 분석 가능

        1. 감기(명사:cold) + 는(조사)
        2. 감(동사 어간) + 기(명사화 어미) + 는(조사)
        3. 감(동사 어간) + 기는(어미)

- 접두사, 접미사 처리
- 고유명사, 사전에 등록되지 않은 단어 처리
특히 띄어쓰기가 없으면 더욱 어려워짐.

<br>

> NLP Tasks

- Word Segmentation (단어 분리)
- Morphological Analysis (형태소 분석)
- Part Of Speech(POS) Tagging (품사태깅)

### 2. 구문 분석 (Syntax Analysis)

**: 문자의 구조를 분석** = 파싱(parsing)

1. 입력 문장에 대한 **문법 구조**를 분석 => **구문 트리** 로 나타냄

![syntax-tree](https://user-images.githubusercontent.com/35002380/73628765-804f3480-4694-11ea-9962-348396ff47e0.jpg)

2. 문장이 **문법적으로 옳은가**를 판단 : concerns the **proper ordering of words**
<br>

        - The dog bit the boy. (O)
        - The boy bit the dog. (O)
        - Bit boy dog the the. (X)

<br>

> 구문 분석의 난점

**Structural Ambiguities 구조적 중의성**

하나의 문장이 다수의 구조로 해석될 수 있는 성질.
구조적 중의성 때문에 각각의 파서마다 구문분석의 결과가 달라질 수 있음.

        John saw Mary in the park
        - John이 Mary를 공원에서 봤다.
        - John이 공원에 있는 Mary를 봤다.

<br>

> NLP Tasks

- Phrase Chunking (간단하게 문장 잘라주는 것)
- Dependency parsing (구문 분석)

### 3. 의미 분석 (Semantic Analysis)

**: 구문 분석 결과 생성된 통사 구조에 의해 문장의 의미를 밝혀내는 작업**.
Concerns the **meaning of** words, phrases, and sentences.

동형이의어, 동음이의어, 다의어의 의미를 정확히 파악하여 문장 전체의 의미 이해
*같은 syntax tree 안에 있는 단서를 가지고 여러 동형이의어/동음이의어/다의어 중에 의미가 맞는 경우의 의미를 채택*

1. 의미적으로 옳은 문장인지를 판별

        - 사람이 사과를 먹는다. (O)
        - 사람이 비행기를 먹는다. (X)
        - 비행기가 사과를 먹는다. (X)

2. 문장 내 단어의 중의성 해소, 생략된 표현이나 대명사 등이 무엇을 지시하는가 등을 파악

=> 구문 분석 결과가 갖는 한계를 해소하기 위함.

<br>

> NLP Tasks

- Word Sense Disambiguation(WSD)
- Semantic Role Labeling (SRL)
- Semantic Parsing
- Textual Entailment

### 4. 화용 분석 (Pragmatic Analysis)

**: 언어의 사용에 관련된 지식을 통해 문장을 해석함으로써 화자의 의도를 파악하는 작업**

        오늘은 비가 옵니다. - 사실전달 vs 우산을 들고 나가라

<br>

> NLP Tasks

- Co-reference/Anaphora Resolution (대명사의 지시 대상)
- Ellipsis Resolution

## 자연어 처리와 기계학습

### 자연어 처리의 접근 방법

1. **언어학적 분석** : 규칙, 사전에 의존
2. **통계적 분석** : 경험적. 대규모 라벨링 데이터로부터 관련 데이터 추출
<= 기계학습이 이용됨

### 기계학습 적용

지속적으로 추가되는 어휘, 많은 오류들, 새로 생기는 개체명이나 전문용어를 규칙에 의한 접근(언어학적 분석)만으로는 따라잡을 수 없음.

기계학습을 통해

- 단어의 **중의성** 해소
    - 품사의 중의성(형태소 분석)
    - 구문구조의 중의성(구문분석)
    - 다의어의 의미적 중의성(의미분석)
    - 대명사의 참조 선택(화용분석)
- 주변 문맥을 보고 자연언어의 모호성을 없애는 단서 인지
    - 선택에 도움이 되는 **특성**이나 **패턴** 추출

와 같은 문제 해결. (분류 문제)

=> 벡터공간 모델 => Word Embedding
