# Kaggle
# Child Mind Institute — Problematic Internet Use (PIU)
https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use

---
## 기술 스택 (Tech Stack)

* **Language**: Python
* **ML Models**: XGBoost, LightGBM, Catboost
* **Hyperparameter Tuning**: Optuna
* **Feature Selection**: SHAP
* **Imputation**: KNNImputer (Scikit-learn)
* **Data Manipulation & Analysis:** Pandas, NumPy, Polars
* **Scientific Computing:** SciPy
* **Visualization**: Matplotlib, Seaborn, Optuna Dashboard
* **Cross-validation**: StratifiedKFold (Scikit-learn)
* **OS**: Linux (Ubuntu Desktop 24.04 LTS)
* **IDE**: VSCode, Jupyter Notebook

---

# 한국어

## 1. 프로젝트 개요

본 프로젝트에서는 Child Mind Institute에서 주최한 Kaggle 경진대회에 참여하여 아동 및 청소년의 인구 통계 신체 데이터, 설문, 액티그래피 측정 기록 데이터 등를 활용하여 문제성 인터넷 사용(PIU)을 예측하는 머신러닝 모델을 개발했습니다.

단독으로 참여하였고, 은메달(상위 1.5%)을 수상하였습니다.

청소년들의 PIU(잠재적 인터넷 중독) 위험을 식별하는 강력한 분류 모델을 구축하여, 조기 개입 전략을 세우고 정신 건강 결과를 개선하는 것이 주된 목표였습니다.

## 2. 문제

문제성 인터넷 사용(PIU)은 아동 및 청소년의 정신 건강과 일상 기능에 영향을 미치는 문제이며, 이에 대해 커져가는 우려가 있습니다. PIU의 조기 식별은 효과적인 개입을 위해 매우 중요합니다. 본 프로젝트는 활동량 기록(actigraphy), 인구 통계 및 행동 데이터와 같은 여러 데이터 소스의 복합적인 특징을 분석하여 PIU를 정확하게 분류하는 예측 모델을 만드는 것을 목표로 했습니다.

## 3. 데이터셋

캐글을 통해 Child Mind Institute가 제공한 데이터셋은 아래를 포함합니다:

    Demographics - 참여자의 나이와 성별.
    Internet Use - 매일 컴퓨터/인터넷을 사용하는 시간.
    Children's Global Assessment Scale - 정신 건강 임상의들이 사용하는 18세 미만 청소년들의 일상 생활을 평가하는 척도.
    Physical Measures - 혈압, 심박수, 키, 몸무게, 허리 둘레, 엉덩이 둘레.
    FitnessGram Vitals and Treadmill - NHANES treadmill protocol을 이용한 심혈관 건강 측정도.
    FitnessGram Child - 건강과 관련된 다섯가지 측정: 유산소 운동 능력치, 근력, 근지구력, 유연성, 체성분.
    Bio-electric Impedance Analysis - BMI, 지방, 근육, 체수분 등을 포함한 주요 신체 측정.
    Physical Activity Questionnaire - 마지막 7일간 아동의 고강도 신체활동 참여 정보.
    Sleep Disturbance Scale - 아동의 수면 장애 척도.
    Actigraphy - 연구용 바이오트래커를 통한 일상적인 신체 활동의 객관적인 측정.
    Parent-Child Internet Addiction Test - 강박성, 도피, 의존성 등 인터넷의 강박적인 사용과 관련된 특성 및 행동을 측정하는 20개 항목 척도.

    ### 액티그래피 파일 및 칼럼 설명 ###
    HBN 연구에 참여하는 동안 일부 참가자들은 가정에서 일상생활을 하는 동안 최대 30일간 지속적으로 착용할 수 있는 가속도계를 지급받았습니다.

    series_{train|test}.parquet/id={id} - 각 시리즈는 여러 날에 걸쳐 한 명의 피험자에 대한 연속적인 가속도계 데이터 기록이며, ID별로 분할되어 훈련 데이터로 사용됩니다.
    id - train/test.csv 파일의 id 필드에 해당하는 환자 식별자.
    step - 각 시리즈 내 개별 관측치에 대한 정수 타임스탬프.
    X, Y, Z - 손목시계형 기기가 각 표준 축을 따라 경험하는 가속도(단위: g) 측정치.
    enmo - wristpy 패키지에 의해 계산 및 설명된 ENMO는 모든 가속도계 신호(x, y, z축 각각, g-force 단위로 측정)의 유클리드 노름 마이너스 원(Euclidean Norm Minus One)이며, 음수 값은 0으로 반올림됩니다. 0 값은 움직임이 없는 기간을 나타냅니다. 이 분야에 표준 가속도 측정 기준이 존재하지는 않지만, 이는 일반적으로 계산되는 여러 특징 중 하나입니다.
    anglez - wristpy 패키지에 의해 계산 및 설명된 Angle-Z는 개별 가속도계 구성 요소에서 파생된 지표로, 팔이 수평면에 대해 이루는 각도를 나타냅니다.
    non-wear_flag - 가속도계 데이터의 표준 편차와 범위를 사용하는 GGIR 정의에 따라, 시계가 제거된 기간을 파악하는 데 도움이 되는 플래그(0: 시계 착용 중, 1: 시계 미착용 중).
    light - 주변 광량을 룩스(lux) 단위로 측정한 값.
    battery_voltage - 배터리 전압을 밀리볼트(mV) 단위로 측정한 값.
    time_of_day - 데이터가 샘플링된 5초 구간의 시작을 나타내는 시간이며, 형식은 %H:%M:%S.%9f 입니다.
    weekday - 요일을 정수로 코딩한 것으로, 1은 월요일, 7은 일요일입니다.
    quarter - 연중 분기. 1부터 4까지의 정수입니다.
    relative_date_PCIAT - PCIAT(문제성 컴퓨터/인터넷 사용 평가 도구) 테스트가 시행된 이후 경과된 일수(정수). 음수는 활동량 측정 데이터가 테스트 시행 이전에 수집되었음을 나타냅니다.

    * **목표 변수: 심각도 손상 지수(SII)**는 문제성 인터넷 사용(PIU) 수준을 식별하는 다중 클래스 분류입니다.


## 4. 방법론 및 접근 방식

본 프로젝트는 포괄적인 머신러닝 파이프라인을 따랐습니다:

1.  **데이터 로드 및 초기 탐색:**
    * 다양한 데이터 형식(활동량 측정 데이터는 parquet, 표 형식 데이터는 CSV)을 로드하고 이해했습니다.
    * 초기 건전성 검사(sanity checks)와 기본적인 통계 분석을 수행했습니다.

2.  **데이터 전처리 및 특징 공학 주요 내용:**
    * **활동량 측정 데이터:** 다양하고 독특한 데이터 집계 방식에서 강력한 통계적 특징들을 추출했습니다.
    * **표 형식 데이터:** 결측값을 처리하고, 범주형 특성을 인코딩하며, 의미 있는 특성(features)을 엔지니어링했습니다.
    * **데이터 통합:** 활동량 측정 데이터에서 추출한 특징들과 인구 통계 및 행동 데이터를 모델 훈련을 위한 통합된 데이터셋으로 결합하기 위한 전략을 개발했습니다.

3.  **모델 선택 및 아키텍처:**
    * 분류에 적합한 다양한 머신러닝 모델을 탐색했습니다.
    * XGBoost, LightGBM, Catboost와 같은 그래디언트 부스팅 결정 트리 모델을 사용하고 앙상블했습니다.
    * 이 문제는 분류 문제이지만, 목표 변수가 순서형이기 때문에 회귀 모델을 사용했습니다. 예측값은 SciPy의 scipy.optimize.minimize 메서드를 사용하여 최적의 임계값을 찾아 클래스로 후처리되었습니다.

4.  **훈련 및 검증:**
    * StratifiedKFold를 활용하여 모델의 일반화를 보장했습니다.
    * RMSE 손실 및 **Quadratic Weighted Kappa(QWK)**와 같은 지표를 사용하여 훈련 진행 상황을 모니터링했습니다.
    * 과적합을 방지하기 위한 기술(예: 드롭아웃, 조기 종료, L1/L2 정규화)을 적용했습니다.
  
5.  **하이퍼파라미터 튜닝 및 특징 선택:**
    * 자동화된 하이퍼파라미터 탐색을 위한 최첨단 오픈 소스 라이브러리인 Optuna를 사용하여 하이퍼파라미터 튜닝을 진행했습니다. https://optuna.org/
    * 예시 이미지들:
        * ![Optuna Dashboard](<images/Screenshot from 2025-06-08 17-43-57.png>)
        * ![Optuna hyperparameter tuning 1](<images/learning_rate_optuna_visualization.png>)
        * ![Optuna hyperparameter tuning 2](<images/max_depth_optuna_visualization.png>)
        * ![Optuna hyperparameter tuning 3](<images/subsample_optuna_visualization.png>)
    * 머신러닝 모델 결과 설명을 위한 최첨단 오픈 소스 라이브러리인 **SHAP(SHapley Additive exPlanations)**을 특징 선택에 사용했습니다. https://shap.readthedocs.io/
        * ![SHAP feature importance](<images/feature_importance.png>)

6.  **모델 평가:**
    * 최종 모델의 성능을 주된 평가 지표인 **QWK(Quadratic Weighted Kappa)**를 사용하여 한 번도 보지 못한 테스트 세트로 평가했습니다.
    * "SHAP을 사용하여 어떤 특징들이 예측에 가장 많이 기여했는지 이해하기 위한 해석 가능성 분석을 수행했습니다.

## 5. 결과 및 주요 관찰

* **달성한 QWK 점수:** 제 모델은 최종 테스트 세트에서 0.461의 QWK 점수를 달성하여 준수한 성능을 보였습니다. 그러나 대회 1위 모델의 점수가 0.482인 점을 고려할 때, 주어진 데이터셋으로는 더 이상의 의미 있는 성능 개선은 어려워 보입니다.
* **[핵심 관찰 1]:** 활동량 측정 데이터의 통합은 모델 성능을 의미 있게 향상시켰으며, 이는 인구 통계학적 특성과의 시너지 효과를 시사합니다.
* **[핵심 관찰 2]:** 데이터셋에 결측값이 많고 가속도계를 착용한 참가자 수가 제한적이었기 때문에, 준수한 모델 성능은 예상했던 바입니다. 더 나은 품질의 데이터가 있다면, 고성능 모델을 만드는 것은 분명히 가능해 보입니다.
* **모델 성능 시각화 예시 이미지:**
    * ![Training and validation loss & validation QWK score](<images/Screenshot from 2025-06-08 19-49-23.png>)
* 심층 학습 모델(TabNet)은 작동하지 않았습니다.

## 6. 결론 및 향후 과제

본 프로젝트는 문제성 인터넷 사용(PIU) 예측 모델을 성공적으로 개발하여, 복잡한 데이터를 다루고 분류를 위해 머신러닝 기술을 적용하는 능숙함을 보여주었습니다. 얻어진 관찰은 정신 건강 진단에 있어 데이터 기반 접근 방식의 잠재력을 부각시킵니다.

**향후 개선 사항:**
* 전이 학습(transfer learning) 또는 데이터 증강(data augmentation)을 위한 외부 데이터셋 통합.
* PIU의 핵심 지표를 식별하기 위해 보다 심층적인 특징 중요도 분석 수행.
* 임상 적용을 위해 모델 예측을 더욱 해석 가능하게 만드는 설명 가능한 AI(XAI) 기술 조사.
* CV 점수(0.529)와 LB 점수(0.461) 간의 현저한 차이를 조사하는 것. 이는 과적합을 나타낼 수 있습니다.
* 더 나은 앙상블 기법 적용.

## 7. 프로젝트 실행 방법

분석 및 모델 훈련 재현 방법:

1.  **Repository 복제:**
    ```bash
    git clone [https://github.com/maxthenobody/kaggle-childmind.git](https://github.com/maxthenobody/kaggle-childmind.git)
    cd kaggle-childmind
    ```
2.  **데이터셋 다운로드:**
    * 캐글에서 대회에 참가하세요. [Child Mind Institute Problematic Internet Use competition](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use)
    * 데이터를 다운받은 후 알맞은 디렉토리에 저장하세요.
3.  **가상 환경을 생성하고 필요한 라이브러리들을 설치해주세요:**
    ```bash
    conda create -n childmind python=3.12 # or venv
    conda activate childmind
    pip install -r requirements.txt
    ```
4.  **Jupyter Notebook을 실행해주세요:**
    ```bash
    jupyter notebook "Kaggle Childmind Project.ipynb"
    ```
    데이터 처리, 모델 훈련 및 평가를 실행하려면 노트북의 단계를 따르세요.

---

# English

## 1. Project Overview

This project addresses the **Child Mind Institute Problematic Internet Use (PIU)** competition on Kaggle, focusing on developing machine learning models to predict problematic internet use in children and adolescents. The solution leverages a diverse dataset comprising demographic, behavioural and actigraphy data.

I participated in the competition alone and won a silver medal (top 1.5%).

The primary goal was to build a robust classification model capable of identifying individuals at risk of PIU, contributing to early intervention strategies and improved mental health outcomes for young people.

## 2. Problem Statement

Problematic Internet Use (PIU) is a growing concern, impacting mental health and daily functioning in children and adolescents. Early identification of PIU is crucial for effective intervention. This project aimed to create a predictive model by analyzing complex features from multiple data sources such as actigraphy, demographic and behavioral data to accurately classify individuals with PIU.

## 3. Dataset

The dataset provided by the Child Mind Institute (via Kaggle) includes:

    Demographics - Information about age and sex of participants.
    Internet Use - Number of hours of using computer/internet per day.
    Children's Global Assessment Scale - Numeric scale used by mental health clinicians to rate the general functioning of youths under the age of 18.
    Physical Measures - Collection of blood pressure, heart rate, height, weight and waist, and hip measurements.
    FitnessGram Vitals and Treadmill - Measurements of cardiovascular fitness assessed using the NHANES treadmill protocol.
    FitnessGram Child - Health related physical fitness assessment measuring five different parameters including aerobic capacity, muscular strength, muscular endurance, flexibility, and body composition.
    Bio-electric Impedance Analysis - Measure of key body composition elements, including BMI, fat, muscle, and water content.
    Physical Activity Questionnaire - Information about children's participation in vigorous activities over the last 7 days.
    Sleep Disturbance Scale - Scale to categorize sleep disorders in children.
    Actigraphy - Objective measure of ecological physical activity through a research-grade biotracker.
    Parent-Child Internet Addiction Test - 20-item scale that measures characteristics and behaviors associated with compulsive use of the Internet including compulsivity, escapism, and dependency.

    ### Actigraphy Files and Field Descriptions ###
    During their participation in the HBN study, some participants were given an accelerometer to wear for up to 30 days continually while at home and going about their regular daily lives.

    series_{train|test}.parquet/id={id} - Series to be used as training data, partitioned by id. Each series is a continuous recording of accelerometer data for a single subject spanning many days.
    id - The patient identifier corresponding to the id field in train/test.csv.
    step - An integer timestep for each observation within a series.
    X, Y, Z - Measure of acceleration, in g, experienced by the wrist-worn watch along each standard axis.
    enmo - As calculated and described by the wristpy package, ENMO is the Euclidean Norm Minus One of all accelerometer signals (along each of the x-, y-, and z-axis, measured in g-force) with negative values rounded to zero. Zero values are indicative of periods of no motion. While no standard measure of acceleration exists in this space, this is one of the several commonly computed features.
    anglez - As calculated and described by the wristpy package, Angle-Z is a metric derived from individual accelerometer components and refers to the angle of the arm relative to the horizontal plane.
    non-wear_flag - A flag (0: watch is being worn, 1: the watch is not worn) to help determine periods when the watch has been removed, based on the GGIR definition, which uses the standard deviation and range of the accelerometer data.
    light - Measure of ambient light in lux. See ​​here for details.
    battery_voltage - A measure of the battery voltage in mV.
    time_of_day - Time of day representing the start of a 5s window that the data has been sampled over, with format %H:%M:%S.%9f.
    weekday - The day of the week, coded as an integer with 1 being Monday and 7 being Sunday.
    quarter - The quarter of the year, an integer from 1 to 4.
    relative_date_PCIAT - The number of days (integer) since the PCIAT test was administered (negative days indicate that the actigraphy data has been collected before the test was administered).

    * **Target Variable: Severity Impairment Index (SII)** Multiclass classification identifying the level of Problematic Internet Use (PIU).


## 4. Methodology & Approach

This project followed a comprehensive machine learning pipeline:

1.  **Data Loading & Initial Exploration:**
    * Loaded and understood the various data formats (parquet for actigraphy, CSV for tabular data).
    * Performed initial sanity checks and basic statistical analyses.

2.  **Data Preprocessing & Feature Engineering Highlights:**
    * **Actigraphy Data:** Extracted robust statistical features from various and unique data aggregation.
    * **Tabular Data:** Handled missing values, encoded categorical features, and engineered meaningful features.
    * **Data Integration:** Developed strategies to combine features extracted from actigraphy with demographic and behavioral data into a unified dataset for model training.

3.  **Model Selection & Architecture:**
    * Explored various machine learning models suitable for classification.
    * Gradient boosting decision tree models such as XGBoost, LightGBM, Catboost were used and ensembled.
    * Even though it's a classification problem, the target variable is ordinal, hence regression models were used and the predictions were post-processed into classes using SciPy's scipy.optimize.minimize method which finds optimal thresholds.

4.  **Training & Validation:**
    * Utilized StratifiedKFold to ensure model generalization.
    * Monitored training progress using metrics like RMSE loss and Quadratic Weighted Kappa(QWK).
    * Applied techniques to prevent overfitting (e.g., dropout, early stopping, L1/L2 regularization).
  
5.  **Hyperparameter Tuning & Feature Selection:**
    * Used Optuna for hyperparameter tuning, a state-of-the-art open source library for automated hyperparameter search. https://optuna.org/
    * Example Images:
        * ![Optuna Dashboard](<images/Screenshot from 2025-06-08 17-43-57.png>)
        * ![Optuna hyperparameter tuning 1](<images/learning_rate_optuna_visualization.png>)
        * ![Optuna hyperparameter tuning 2](<images/max_depth_optuna_visualization.png>)
        * ![Optuna hyperparameter tuning 3](<images/subsample_optuna_visualization.png>)
    * Used SHAP(SHapley Additive exPlanations) for feature selection, a state-of-the-art open source library for explaining machine learning model outputs. https://shap.readthedocs.io/
        * ![SHAP feature importance](<images/feature_importance.png>)

6.  **Model Evaluation:**
    * Evaluated the final model performance on an unseen test set using QWK, the primary competition metric.
    * Performed interpretability analyses to understand which features contributed most to predictions, using SHAP.

## 5. Results & Key Findings

* **Achieved QWK Score:** My model achieved a QWK score of **0.461** on the final test set, demonstrating moderate performance. However, considering the #1 winner of the competition's score is 0.482, it seems meaningful improvements are not feasible with the given dataset.
* **[My Key Insight 1]:** The integration of actigraphy data meaningfully improved model performance, suggesting a synergistic effect with demographic features.
* **[My Key Insight 2]:** Given the high proportion of null values in the dataset and only a limited number of participants were given an accelerometer, moderate model performance was expected. With better quality data, high performing model creation seems certainly achievable.
* **Model Performance:**
    * ![Training and validation loss & validation QWK score](<images/Screenshot from 2025-06-08 19-49-23.png>)
* Deep learning model (TabNet) didn't work.

## 6. Conclusion & Future Work

This project successfully developed a predictive model for Problematic Internet Use, demonstrating proficiency in handling complex data and applying machine learning techniques for classification. The insights gained highlight the potential of data-driven approaches in mental health diagnostics.

**Future Enhancements could include:**
* Integrating external datasets for transfer learning or data augmentation.
* Conducting more in-depth feature importance analysis to identify critical indicators of PIU.
* Investigating explainable AI (XAI) techniques to make model predictions more interpretable for clinical applications.
* Investigating the noticeable difference between CV score (0.529) and LB score (0.461) which may indicate overfitting.
* Applying better ensemble techniques.

## 7. How to Run This Project

To replicate the analysis and model training:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/maxthenobody/kaggle-childmind.git](https://github.com/maxthenobody/kaggle-childmind.git)
    cd kaggle-childmind
    ```
2.  **Download the dataset:**
    * You will need to join the [Child Mind Institute Problematic Internet Use competition](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use) on Kaggle.
    * Download the data and place it in the appropriate directories.
3.  **Create a virtual environment and install dependencies:**
    ```bash
    conda create -n childmind python=3.12 # or venv
    conda activate childmind
    pip install -r requirements.txt
    ```
4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook "Kaggle Childmind Project.ipynb"
    ```
    Follow the steps in the notebook to execute the data processing, model training, and evaluation.










