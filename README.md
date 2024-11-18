# CTIS471_Term_Project
## Türkçe

Bu repo, Bilkent Üniversitesi CTIS Bölümü CTIS471 kodlu ders kapsamında geliştirilmiş bir veri analizi ve modelleme uygulamasıdır. Python programlama dili kullanılarak, çeşitli makine öğrenimi algoritmaları, veri işleme adımları ve görselleştirme teknikleri üzerine odaklanmaktadır.

## Proje Yapısı

### 1. `Models/`
Yüksek Accuracy alınan makine öğrenimi modellerini içeren dosya  :
- **`Model_ANN_CV.py`**: Yapay sinir ağları (ANN) ve çapraz doğrulama.
- **`Model_MLP_CV.py`**: Çok katmanlı algılayıcı (MLP) ve çapraz doğrulama.
- **`model_RF_CV.py`**: Rastgele orman (RF) algoritması ve çapraz doğrulama.

### 2. `Single-Pipeline/`
Makine öğrenimi süreçlerini tek bir pipeline içinde birleştiren yapı:
- **`pipeline.py`**: Veri ön işleme ve modelleme işlemlerinin tüm aşamalarını içeren Python betiği.

### 3. `Step-Pipeline/`
Pipeline işlemini adım adım gerçekleştiren dosyalar:
- **`1_metric_manipulation.py`**: Modelleme sürecinde metrik manipülasyonları gerçekleştiren adım.
- **`2_anomaly_removal.py`**: Anomali tespiti ve kaldırılması için kullanılan adım.
- **`3_dataset_prep.py`**: Veri setinin işlenmesi ve modellemeye hazırlanması.
- **`kickstarter.py`**: Step-Pipeline içinde yer alan Python betiklerini sırasıyla tetikleyen betik.

### 4. `Unused-Models/`
Henüz kullanılmayan veya alternatif olarak tutulan modeller:
- **`Model_KNN_CV.py`**: K-en yakın komşu (KNN) modeli ve çapraz doğrulama.
- **`Model_Logistic_Regression_CV.py`**: Lojistik regresyon modelive çapraz doğrulama.
- **`Model_SVM_CV.py`**: Destek vektör makineleri (SVM) modeli ve çapraz doğrulama.

### 5. `Visualization/`
Veri görselleştirme için kullanılan Python betikleri:
- **`Correlation_Matrix.py`**: Korelasyon matrisi oluşturma ve görselleştirme.
- **`Feature_Importance.py`**: Özellik önemini görselleştirme.
- **`RFE.py`**: Recursive Feature Elimination (RFE) yöntemi.

---

## Veri Seti

- **`CTIS471_2425_first_part.csv`**: Projede kullanılan veri seti.

---

## Kurulum

## Kurulum

Bu projeyi çalıştırmak için aşağıdaki adımları takip edin:

1. **Gerekli Kütüphaneler:**
   Projede kullanılan Python kütüphanelerini kurmak için aşağıdaki komutları çalıştırabilirsiniz:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn tensorflow scikeras

2. **Proje Dizini:** 
    Proje dizin yapısını doğru bir şekilde koruduğunuzdan emin olun.

3. **TensorFlow ile İlgili Ekstra Gereksinimler:** 
    Eğer TensorFlow kütüphanesi GPU desteği ile çalıştırılacaksa, uygun CUDA ve cuDNN sürümlerinin yüklü olduğundan emin olun. 
    İlgili dökümana TensorFlow Kurulum Kılavuzu üzerinden ulaşabilirsiniz.

4. **Başlangıç Testi:** 
    Projenin düzgün kurulumunu test etmek için aşağıdaki komutları çalıştırarak kütüphanelerin doğru yüklendiğini doğrulayabilirsiniz:
    ```bash
    python -c "import pandas; import numpy; import sklearn; import matplotlib; import seaborn; import tensorflow; print('Kurulum başarılı!')"

-----

# CTIS471_Term_Project
## English

This repository contains a data analysis and modeling application developed as part of the CTIS471 course in the Department of CTIS at Bilkent University. It focuses on various machine learning algorithms, data processing steps, and visualization techniques using Python programming language.

## Project Structure

### 1. `Models/`
Files containing machine learning models with high accuracy:
- **`Model_ANN_CV.py`**: Artificial Neural Networks (ANN) and cross-validation.
- **`Model_MLP_CV.py`**: Multi-Layer Perceptron (MLP) and cross-validation.
- **`model_RF_CV.py`**: Random Forest (RF) algorithm and cross-validation.

### 2. `Single-Pipeline/`
A structure that combines machine learning processes into a single pipeline:
- **`pipeline.py`**: Python script that includes all stages of data preprocessing and modeling.

### 3. `Step-Pipeline/`
Files that execute the pipeline step-by-step:
- **`1_metric_manipulation.py`**: Step for metric manipulation during the modeling process.
- **`2_anomaly_removal.py`**: Step for anomaly detection and removal.
- **`3_dataset_prep.py`**: Step for processing the dataset and preparing it for modeling.
- **`kickstarter.py`**: Script that sequentially triggers the Python scripts within the Step-Pipeline.

### 4. `Unused-Models/`
Models that are currently unused or kept as alternatives:
- **`Model_KNN_CV.py`**: K-Nearest Neighbors (KNN) model and cross-validation.
- **`Model_Logistic_Regression_CV.py`**: Logistic Regression model and cross-validation.
- **`Model_SVM_CV.py`**: Support Vector Machines (SVM) model and cross-validation.

### 5. `Visualization/`
Python scripts used for data visualization:
- **`Correlation_Matrix.py`**: Creates and visualizes a correlation matrix.
- **`Feature_Importance.py`**: Visualizes feature importance.
- **`RFE.py`**: Recursive Feature Elimination (RFE) method.

---

## Dataset

- **`CTIS471_2425_first_part.csv`**: Dataset used in the project.

---

## Setup

Follow the steps below to run this project:

1. **Required Libraries:**  
   Install the Python libraries used in the project by running the following command:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn tensorflow scikeras
    ```

2. **Project Directory:**  
   Ensure that the project directory structure is maintained correctly.

3. **Additional Requirements for TensorFlow:**  
   If TensorFlow is to be run with GPU support, ensure the appropriate versions of CUDA and cuDNN are installed.  
   You can refer to the TensorFlow Installation Guide for more details.

4. **Initial Test:**  
   Test the proper setup of the project by running the following commands to verify that the libraries are installed correctly:
    ```bash
    python -c "import pandas; import numpy; import sklearn; import matplotlib; import seaborn; import tensorflow; print('Setup Successful!')"
    ```
