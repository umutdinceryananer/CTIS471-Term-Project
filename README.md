# CTIS471_TP_1

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
