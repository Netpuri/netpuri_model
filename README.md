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

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6483300e-7c1d-4fd3-ae01-9d3e0d35a9c2/7e01a3e3-0554-45d6-b50b-ca1b217b1b21/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6483300e-7c1d-4fd3-ae01-9d3e0d35a9c2/3aaa93b0-ca0d-4263-b9a7-cc67cd9070fa/image.png)

- **결과 분석**
    - 추가된 데이터로 인해 3번 클래스의 성능이 약간 향상되었으나, 여전히 1번(정치성 글)으로의 오분류가 많았다.
    - 그러나 사용성을 생각했을 때 해당이 없는 데이터를 오인식하는 것은 매우 불편하게 느껴질 것이라고 판단하였다.

## 4. Hazard Filter 모델

- **모델 구성**
    - 사용자가 정상 댓글을 유해 댓글로 오분류하는 경우 불편을 느낄 수 있다는 점을 고려하여, 1번 타입(정치성 글)을 제거하고 0(정상)과 1(유해)로 나머지 댓글을 분류하는 Hazard Filter 모델을 개발했다.
- **학습 결과**
    - Hazard Filter 모델은 정상 댓글을 최대한 정확하게 분류하면서 유해 댓글을 탐지하는 데 중점을 두어 학습되었다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6483300e-7c1d-4fd3-ae01-9d3e0d35a9c2/2a813336-4bc3-4550-b226-1e4f2635cceb/image.png)

## 5. 최종 구조

- **1차 필터링: Hazard Filter 모델**
    - 댓글의 유해성을 1차적으로 필터링
- **2차 필터링: Type Filter 모델**
    - Hazard Filter 모델에서 유해한 것으로 분류된 댓글을 다시 세부적으로 유형(일반, 정치, 성적, 우울, 공격적)으로 분류

이와 같은 구조로 최종 필터링 시스템을 구축하여 인터넷상에서 부정적인 영향을 미칠 수 있는 다양한 유형의 정보를 효율적으로 필터링할 수 있게 함.

## License and Acknowledgments

This project is licensed under the Apache License 2.0. You may not use this project except in compliance with the License. A copy of the License is included in this repository under the `LICENSE` file.

This project uses several open-source libraries and datasets. Below are the licenses and acknowledgments for each:

- **Pandas**: Licensed under the BSD 3-Clause License.
- **Scikit-learn**: Licensed under the BSD 3-Clause License.
- **PyTorch**: Licensed under the BSD 3-Clause License.
- **Transformers (Hugging Face)**: Licensed under the Apache License 2.0.
- **Imbalanced-learn**: Licensed under the BSD 3-Clause License.
- **NLP-Aug**: Licensed under the MIT License.
- **KcELECTRA-base (Beomi)**: Licensed under the MIT License.
- **YouTube API**: This project complies with the YouTube API Services Terms of Service.
- **AIHub Datasets**: This project uses datasets provided by AIHub, in compliance with AIHub's terms of use.
- **Korean Hate Speech Dataset**: This project uses the Korean Hate Speech Dataset, in compliance with the dataset's terms of use.
- **한국어 감정 정보가 포함된 단발성 대화 데이터셋**: This project uses the dataset provided by AIHub, in compliance with AIHub's terms of use.
- **텍스트 윤리 검증 데이터셋**: This project uses the dataset provided by AIHub, in compliance with AIHub's terms of use.
- **감성 대화 말뭉치**: This project uses the dataset provided by AIHub, in compliance with AIHub's terms of use.
