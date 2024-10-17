# 텍스트 필터링 모델

## 1. 배경

인터넷 접근성이 향상되면서 정보에 대한 접근이 쉬워졌으나, 부정적인 영향을 미칠 수 있는 정보도 많이 생성되고 있다.

특히, 디지털 격차로 인해 인터넷 활용 능력이 부족한 사람들은 이러한 정보에 더 큰 영향을 받을 수 있다.

이를 해결하기 위해 텍스트 필터링 모델을 개발하고자 하였다.

## 2. 데이터 수집 및 가공

### 2.1 데이터 타입 분류

- 일반 0
- 정치성 글 1
- 성적인 글 2
- 우울한 글 3 (슬픔)
- 공격적인 글 4 (분노, 혐오)

### 2.2 초기 데이터셋 구성

- Korean Hate Speech Dataset
- https://github.com/kocohub/korean-hate-speech
    - 분류 기준: `contain_gender_bias`, `bias`, `hate`
    - 분류 방식: `false, none, none`인 경우 0번(해당 없음), 나머지는 4번(공격적인 글)으로 분류
    - 총합 결과:
        - 0: 3429개
        - 4: 4945개
- **한국어 감정 정보가 포함된 단발성 대화 데이터셋**
- https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=270
    - 분류 기준: 감정 유형에 따라 `0(해당 없음)`, `3(우울한 글)`, `4(공격적인 글)`로 분류
    - 총합 결과:
        - 0: 22233개
        - 3: 5267개
        - 4: 16039개
- **YouTube API**
    - 정치성 댓글과 성희롱 댓글을 수집하여 각각 `1(정치성 글)`, `2(성적인 글)`로 분류
    - 정치성 댓글 편향을 방지하기 위해 나무위키를 활용하여 구독자가 가장 많은 진보, 보수 진영의 유튜버 두명의 동영상에서 댓글을 수집함
        - [https://youtu.be/O7lFQHRoiok?si=rtpUaxxXiobGtYFa](https://youtu.be/O7lFQHRoiok?si=rtpUaxxXiobGtYFa%20https://youtu.be/L50FQAAhIVM?si=0cI_esf4lJ3NuLVu)
        - [https://youtu.be/L50FQAAhIVM?si=0cI_esf4lJ3NuLVu](https://youtu.be/O7lFQHRoiok?si=rtpUaxxXiobGtYFa%20https://youtu.be/L50FQAAhIVM?si=0cI_esf4lJ3NuLVu)
        - https://youtu.be/-eGJjxr1MWU?si=G3waZAHQmmDlcr_H
        - https://youtu.be/CGM-GzS7Wqs?si=_H9C3in1W5L0Sz7V
        - [https://youtu.be/xBPeTy3gxFU?si=nmOJ1nDV7FU7pJGl](https://youtu.be/xBPeTy3gxFU?si=nmOJ1nDV7FU7pJGl%20https://youtu.be/OHtYwkRigy0?si=wJVP7LbNakfD2VKZ)
        - [https://youtu.be/OHtYwkRigy0?si=wJVP7LbNakfD2VKZ](https://youtu.be/xBPeTy3gxFU?si=nmOJ1nDV7FU7pJGl%20https://youtu.be/OHtYwkRigy0?si=wJVP7LbNakfD2VKZ)
        - https://youtu.be/WEgYxZ8eh14?si=OXfzmIYlXeuQxh_t
        - https://youtu.be/LMhXLx_tLro?si=R8XlG6FR5lergsVi
    - 조회수 기준으로 직캠 영상에서 댓글을 수집
        - [https://youtu.be/ljnr5Rnzy-4?si=TsaNQh7xdRckkrEk](https://youtu.be/ljnr5Rnzy-4?si=TsaNQh7xdRckkrEk%20https://youtu.be/mL5xgpUiESg?si=m_g7rEw75RFQrvq7%20https://youtu.be/gZeyLFXrrkE?si=FwnygohbGB5DCj7R%20https://youtu.be/E4T08bHmEmE?si=PpQ-UxX2Kuug6k_q%20https://youtu.be/OpJOUU5rePY?si=6eIEuk1Yu49QkGvK%20https://youtu.be/0rvDDFBoxII?si=qSsJXcNhh99MB9Mo)
        - [https://youtu.be/mL5xgpUiESg?si=m_g7rEw75RFQrvq7](https://youtu.be/ljnr5Rnzy-4?si=TsaNQh7xdRckkrEk%20https://youtu.be/mL5xgpUiESg?si=m_g7rEw75RFQrvq7%20https://youtu.be/gZeyLFXrrkE?si=FwnygohbGB5DCj7R%20https://youtu.be/E4T08bHmEmE?si=PpQ-UxX2Kuug6k_q%20https://youtu.be/OpJOUU5rePY?si=6eIEuk1Yu49QkGvK%20https://youtu.be/0rvDDFBoxII?si=qSsJXcNhh99MB9Mo)
        - [https://youtu.be/gZeyLFXrrkE?si=FwnygohbGB5DCj7R](https://youtu.be/ljnr5Rnzy-4?si=TsaNQh7xdRckkrEk%20https://youtu.be/mL5xgpUiESg?si=m_g7rEw75RFQrvq7%20https://youtu.be/gZeyLFXrrkE?si=FwnygohbGB5DCj7R%20https://youtu.be/E4T08bHmEmE?si=PpQ-UxX2Kuug6k_q%20https://youtu.be/OpJOUU5rePY?si=6eIEuk1Yu49QkGvK%20https://youtu.be/0rvDDFBoxII?si=qSsJXcNhh99MB9Mo)
        - [https://youtu.be/E4T08bHmEmE?si=PpQ-UxX2Kuug6k_q](https://youtu.be/ljnr5Rnzy-4?si=TsaNQh7xdRckkrEk%20https://youtu.be/mL5xgpUiESg?si=m_g7rEw75RFQrvq7%20https://youtu.be/gZeyLFXrrkE?si=FwnygohbGB5DCj7R%20https://youtu.be/E4T08bHmEmE?si=PpQ-UxX2Kuug6k_q%20https://youtu.be/OpJOUU5rePY?si=6eIEuk1Yu49QkGvK%20https://youtu.be/0rvDDFBoxII?si=qSsJXcNhh99MB9Mo)
        - [https://youtu.be/OpJOUU5rePY?si=6eIEuk1Yu49QkGvK](https://youtu.be/ljnr5Rnzy-4?si=TsaNQh7xdRckkrEk%20https://youtu.be/mL5xgpUiESg?si=m_g7rEw75RFQrvq7%20https://youtu.be/gZeyLFXrrkE?si=FwnygohbGB5DCj7R%20https://youtu.be/E4T08bHmEmE?si=PpQ-UxX2Kuug6k_q%20https://youtu.be/OpJOUU5rePY?si=6eIEuk1Yu49QkGvK%20https://youtu.be/0rvDDFBoxII?si=qSsJXcNhh99MB9Mo)
        - [https://youtu.be/0rvDDFBoxII?si=qSsJXcNhh99MB9Mo](https://youtu.be/ljnr5Rnzy-4?si=TsaNQh7xdRckkrEk%20https://youtu.be/mL5xgpUiESg?si=m_g7rEw75RFQrvq7%20https://youtu.be/gZeyLFXrrkE?si=FwnygohbGB5DCj7R%20https://youtu.be/E4T08bHmEmE?si=PpQ-UxX2Kuug6k_q%20https://youtu.be/OpJOUU5rePY?si=6eIEuk1Yu49QkGvK%20https://youtu.be/0rvDDFBoxII?si=qSsJXcNhh99MB9Mo)
    - 총합 결과:
        - 0: 22233개
        - 1: 26778개
        - 2: 643개
        - 3: 5267개
        - 4: 16039개

### 2.3 데이터 추가 수집

- **텍스트 윤리 검증 데이터셋**
- https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=558
    - 분류 기준: `sexual`이 포함된 경우 2번, `immoral_none`인 경우 0번, 나머지는 4번으로 필터링
    - 총합 결과:
        - 0: 188695개
        - 2: 19390개
        - 4: 89080개
- **감성 대화 말뭉치**
- https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=86
    - train 데이터셋의 슬픔에 해당하는 데이터의 사람 문장1을 데이터로 하여 정제함
    - 데이터 9125개 추가
        - 총합 결과
            - 0: 188695개
            - 1: 26778개
            - 2: 19390개
            - 3: 14392개
            - 4: 89080개

## 3. 모델 학습

<img width="310" alt="image" src="https://github.com/user-attachments/assets/d1bffcab-b7d9-459b-9afa-d155361bebee">

<img width="464" alt="image (1)" src="https://github.com/user-attachments/assets/0d27237b-3f4a-4074-97c4-79248f37c07b">


- **결과 분석**
    - 추가된 데이터로 인해 3번 클래스의 성능이 약간 향상되었으나, 여전히 1번(정치성 글)으로의 오분류가 많았다.
    - 그러나 사용성을 생각했을 때 해당이 없는 데이터를 오인식하는 것은 매우 불편하게 느껴질 것이라고 판단하였다.

## 4. Hazard Filter 모델

- **모델 구성**
    - 사용자가 정상 댓글을 유해 댓글로 오분류하는 경우 불편을 느낄 수 있다는 점을 고려하여, 1번 타입(정치성 글)을 제거하고 0(정상)과 1(유해)로 나머지 댓글을 분류하는 Hazard Filter 모델을 개발했다.
- **학습 결과**
    - Hazard Filter 모델은 정상 댓글을 최대한 정확하게 분류하면서 유해 댓글을 탐지하는 데 중점을 두어 학습되었다.

<img width="467" alt="image (2)" src="https://github.com/user-attachments/assets/82409bd0-27c0-4e0c-842c-88dfa85dce77">


## 5. 최종 구조

- **1차 필터링: Hazard Filter 모델**
    - 댓글의 유해성을 1차적으로 필터링
- **2차 필터링: Type Filter 모델**
    - Hazard Filter 모델에서 유해한 것으로 분류된 댓글을 다시 세부적으로 유형(일반, 정치, 성적, 우울, 공격적)으로 분류

이와 같은 구조로 최종 필터링 시스템을 구축하여 인터넷상에서 부정적인 영향을 미칠 수 있는 다양한 유형의 정보를 효율적으로 필터링할 수 있게 함.

이 프로젝트는 다음의 오픈소스 소프트웨어와 데이터셋을 사용하며, 각 라이브러리는 저작권 표시와 함께 해당 라이선스 조건에 따라 배포됩니다:

## 소프트웨어 라이브러리

- Pandas (BSD 3-Clause License)
    
    Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
    All rights reserved.
    
    [BSD 3-Clause License](https://github.com/pandas-dev/pandas/blob/main/LICENSE)
    
- Scikit-learn (BSD 3-Clause License)
    
    Copyright (c) 2007-2024 The scikit-learn developers.
    
    [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)
    
- PyTorch (BSD 3-Clause License)
    
    Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
    
    [BSD 3-Clause License](https://github.com/pytorch/pytorch/blob/master/LICENSE)
    
- Transformers (Hugging Face) (Apache License 2.0)
    
    Copyright 2018- The Hugging Face team. All rights reserved.
    
    [Apache License 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE)
    
- Imbalanced-learn (MIT License)
    
    Copyright (c) 2014-2020 The imbalanced-learn developers.
    All rights reserved.
    
    [MIT License](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/LICENSE)
    
- NLP-Aug (MIT License)
    
    Copyright (c) 2019 Edward Ma
    
    [MIT License](https://github.com/makcedward/nlpaug/blob/master/LICENSE)
    
- KcELECTRA-base (Beomi) (MIT License)
    
    Copyright (c) 2021 Junbum Lee
    
    [MIT License](https://github.com/Beomi/KcELECTRA/blob/master/LICENSE)
    

### API 서비스 및 데이터셋

- YouTube API
    
    이 프로젝트는 YouTube API를 사용하며, YouTube API Services 이용 약관을 준수합니다. [YouTube API Services Terms of Service](https://developers.google.com/youtube/terms/api-services-terms-of-service)
    
- AIHub Datasets
    
    AIHub에서 제공하는 데이터셋을 사용하며, AIHub 이용 약관을 준수합니다. [AIHub 이용 약관](https://aihub.or.kr/legal/terms)
    
- Korean Hate Speech Dataset
    
    데이터셋 이용 약관에 따라 사용하며, 해당 약관을 준수합니다. 각 데이터셋의 공식 홈페이지를 참고하여 이용 약관을 준수하십시오.
    
- 한국어 감정 정보가 포함된 단발성 대화 데이터셋 (AIHub)
    
    AIHub 이용 약관에 따라 사용하며, 해당 약관을 준수합니다. [AIHub 이용 약관](https://aihub.or.kr/legal/terms)
    
- 텍스트 윤리 검증 데이터셋 (AIHub)
    
    AIHub 이용 약관에 따라 사용하며, 해당 약관을 준수합니다. [AIHub 이용 약관](https://aihub.or.kr/legal/terms)
    
- 감성 대화 말뭉치 (AIHub)
    
    AIHub 이용 약관에 따라 사용하며, 해당 약관을 준수합니다. [AIHub 이용 약관](https://aihub.or.kr/legal/terms)
    

각 라이브러리의 라이선스와 고지 사항을 확인하고, 모든 저작권과 라이선스를 준수하여 소프트웨어를 배포하시기 바랍니다. 또한, 데이터셋과 API는 이용 약관에 따라 사용되므로, 이에 대한 이용 약관을 준수하십시오.
