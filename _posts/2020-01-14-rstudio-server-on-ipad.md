---
title: "아이패드에서 Rstudio 사용하기"
last_modified_at: "2020-01-14 17:38:00 +0900"
categories:
  - Tools
tags:
  - RStudio Server
  - iPad
toc: true
toc_sticky: true
---


보통 데이터 분석을 위한 툴로 python, r과 같은 언어들이 많이 사용된다.  
python의 경우 [Carnets](https://apps.apple.com/us/app/carnets/id1450994949), [JUNO](https://apps.apple.com/kr/app/juno-connect-for-jupyter/id1315744137)와 같은 IDE형, 또는 notebook형 어플들이 앱스토어에 등록되어 있는 반면, Rstudio의 경우 관련 어플리케이션을 찾을 수 없었다.  

데이터 분석이나 마크다운 문서를 작성할 때 Rstudio를 많이 사용하는 편인데, 물론 사양 괜찮은 노트북을 가지고 다니는 것이 베스트겠지만, 간단한 분석정도는 굳이 노트북을 들고다니지 않고 iPad로 하고 싶어 방법을 알아보던 중 Rstudio가 Web상에서 접근 가능한 서버 형태로 배포되고 있다는 것을 알게되어 이를 통해 iPad에서 Rstudio를 이용하는 방법을 공유해보고자 한다.  

참고로 작성자는 아이패드 6세대 32GB 기종을 사용하고 있다.

# Rstudio Server on iPad

Rstudio를 iPad에서 사용하기 위해서는 AWS를 이용해 서버 인스턴스를 생성하여, 해당 서버에서 Rstudio Server를 사용해야 한다.  
이 과정에서 RStudio Server에 특화된 AMI(아마존 가상머신 이미지)를 사용할 것이다.  
이후 해당 서버에 대한 인증을 완료하면 iPad에서도 RStudio를 사용할 수 있다.

## AWS를 이용한 서버 인스턴스 생성

![aws](https://user-images.githubusercontent.com/35002380/73628534-dc658900-4693-11ea-8e63-b58e7b66de1d.jpeg)

<br>

[**Rstudio Server**](https://rstudio.com/products/rstudio/download-server/)를 이용하기 위해 Free Tier 요금제(무료)를 제공하고 있는 **아마존 클라우드 서비스(AWS)**를 이용한다.

AWS계정이 없는 경우 [AWS 홈페이지](https://aws.amazon.com/ko/)에서 **Free Tier**요금제를 선택하여 새 계정을 만든다.

<br>

![aws_home](https://user-images.githubusercontent.com/35002380/73628535-dc658900-4693-11ea-97b4-59afba287378.png)

<br>

## AMI(Amazon Machine Image)

이 분야의 능력자들이 RStudio Server를 위한 **아마존 가상머신 이미지(AMI)**를 이미 다 세팅해서 만들어놓았다. 나같이 명령어 치는거 힘들어하고... 컴알못인 사람들은 고마워하면서 이 AMI를 쓰면 된다.

AWS Marketplace에 이미 등록되어 있어 AWS에서 새 인스턴스를 생성할 때 직접 검색하여 사용해도 되지만, 이왕 이 포스팅을 보고 있는거 따로 검색하지 말고 **[Rstudio AMI](http://www.louisaslett.com/RStudio_AMI/)** 왼쪽의 링크를 클릭하도록 한다.

<br>

![ami](https://user-images.githubusercontent.com/35002380/73628533-dbccf280-4693-11ea-9df7-22a3d042cb0a.jpeg)

<br>

*참고로, 작성자는 아이패드 Safari에서 해당 작업을 진행하였을 때는 무슨 이유에선지 되지 않았고, Chrome 웹 브라우저를 사용하였다.*

해당 사이트에서 *Asia Pacific, Seoul* 옆의 링크를 클릭하고 AWS에 로그인하면 자동으로 AMI가 이미 선택되어있다.  
나머지 옵션들은 **기본으로 세팅되어있는대로 넘기면 된다.**

<br>

![inbound](https://user-images.githubusercontent.com/35002380/73628537-dc658900-4693-11ea-9834-4a6004deaf4f.jpeg)

<br>

> *단, 해당 서버는 HTTP 80번 포트를 이용하기 때문에, 6. 보안 그룹 구성 에서 생성한 EC2 Instance에 http로 접근할 수 있도록 하기 위해 HTTP가 80번 포트까지 접근 가능하도록 설정해준다. SSH의 경우 설정을 따로 해주지 않아도 기본적으로 설정이 되어있으나 작성자의 경우 확실하게 하기 위해 다시 설정하였다.*

<br>

![ec2](https://user-images.githubusercontent.com/35002380/73628536-dc658900-4693-11ea-8c4b-71028a8cbdfd.jpeg)

<br>

생성한 EC2 인스턴스의 `설명` 탭에서 `퍼블릭 DNS`에 있는 주소를 주소창에 입력하면 아이디와 패스워드를 입력하는 창이 나타난다.

기본 아이디는 **‘rstudio’**이며, 패스워드는 `설명` 탭에 있던 **`인스턴스 ID`**를 입력하면 된다.  
~~원래 비밀번호도 rstudio였으나 보안 문제로 바뀜. 자세한 사항은 [AMI 다운로드 사이트](http://www.louisaslett.com/RStudio_AMI/) 참고.~~

<br>

![rstudio](https://user-images.githubusercontent.com/35002380/73628538-dcfe1f80-4693-11ea-87c2-086100a38d6f.jpeg)

<br>

위의 과정을 거치면 다음과 같이 아이패드에서도 Rstudio를 이용할 수 있다.

# Rstudio Server on iPad의 한계

iPad에서도 Rstudio를 사용할 수 있다는 장점은 있으나, 기본적으로 태블렛 환경에 적합한 UI가 아니기 때문에 사용하기가 불편하다.

- 일부 단축키 사용 불가
- 특정 panel 스크롤 불가 (화면 전체가 스크롤됨)
- 외장 키보드 사용시 아래가 짤림
- 한글 입력이 아직 완전하지 않음

과 같은 문제들이 있다.

UI와 입력 도구만 개선된다면 iPad에서도 Rstudio를 잘 활용할 수 있지 않을까 하는 의견이다.
