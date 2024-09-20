# Continual learning in KANs

This repository contains the results of the tests carried out to prove the ability of the Kolmogorov-Arnold neural
network to resist the catastrophic forgetting that heavily affects MLPs.

---

### 📘 The results of these tests are presented in detail in this paper:

[KANs Continual Learning [V2] - Morelli Valerio Federica Paganica.pdf](https://github.com/user-attachments/files/15993786/Kolmogorov_Arnold_Networks.pdf)

### 📙 The slideshow presented on the day of the exam:

[KANs Continual Learning [Slideshow PPTX] - Morelli Valerio Paganica Federica.pptx](https://github.com/user-attachments/files/15993699/KANs.Continual.Learning.Morelli.Valerio.Paganica.Federica.pptx)

[KANs Continual Learning [Slideshow PDF] - Morelli Valerio Paganica Federica.pdf](https://github.com/user-attachments/files/15993700/KANs.Continual.Learning.Morelli.Valerio.Paganica.Federica.pdf)

---

<a name="index"></a>

## 📘 Table of Contents

* [📈 Testing different learning rate scales on MLPs and KANs](#LR)
* [⬆️ Sorted MNIST training set (INTRA training set sorting)](#INTRA)
  * [MLPs vs KANs](#INTRA1)
  * [KAN-based and non-KAN-based convolutional nets](#INTRA2)
* [⬇️ Class-IL Scenario (INTER training set sorting)](#INTER)
  * [MLPs vs KANs](#INTER1)
  * [KAN-based and non-KAN-based convolutional nets](#INTER2)
* [❗ The Gaussian Peaks Problem](#EfficientKAN)
* [👨🏻‍💻 Authors](#Authors)

<a name="LR"/></a>

## 📈 Testing different learning rate scales on MLPs and KANs on MNIST

<img width="600rem" src="https://github.com/MrPio/KAN-Continual_Learning_tests/assets/22773005/f375374c-5890-4053-96bf-95ace0dda9bc"/>

<a name="INTRA"/></a>

## ⬆️ Sorted MNIST training set (INTRA training set sorting)

The first test tries to understand the impact of a non-shuffled trainset on the training of the different architectures.
This scenario is common in real-time applications where the order of the input data cannot be decided and the network
may not see the sample of a particular class again.
This is a deliberate attempt to make the network's training less effective because the order of the training sets, as is
well known in machine learning, should always be random in machine learning.
In these unfavourable conditions, the KANs prove to outperform the MLPs

The training set is sorted as follows:

<img width="600rem" src="https://github.com/MrPio/KAN-Continual_Learning_tests/assets/22773005/c7730268-f646-45b9-8587-0e4d742168db"/>

When MLPs see a new, previously unseen digit at the same learning rate, they tend to become distorted more quickly:

<img width="600rem" alt="INTRA dataset lr-5 ep1-2" src="https://github.com/MrPio/KAN-Continual_Learning_tests/assets/22773005/904e3324-47e5-4289-a8a8-694246622f03"/>

The results of this test are:

<a name="INTRA1"/></a>

### MLPs vs KANs

<img width="600rem" alt="INTRAdataset_NON-CONV" src="https://github.com/MrPio/KAN-Continual_Learning_tests/assets/22773005/0dbc7894-a8da-4cd3-9362-1ba48a767974"/>

<a name="INTRA2"/></a>

### KAN-based and non-KAN-based convolutional nets

<img width="600rem" alt="INTRAdataset_CONV2" src="https://github.com/MrPio/KAN-Continual_Learning_tests/assets/22773005/0b731558-8ffe-4576-9f42-f24896eabbec"/>

<a name="INTER"/></a>

## ⬇️ Class-IL Scenario (INTER training set sorting)

🎬 The following video highlights the difference between MLPs and KANs in a Domain-IL scenario:

https://github.com/MrPio/KAN_tests/assets/22773005/b244367a-9af1-4b56-b005-bda6b788d810

<img width="600rem" alt="confusion_matrix" src="https://github.com/MrPio/KAN-Continual_Learning_tests/assets/22773005/4e0561e9-32b8-44d3-ab01-8fd197451940"/>

<a name="INTER1"/></a>

### MLPs vs KANs

Based on ![Convolutional-KANs by Blealtan](https://github.com/Blealtan/efficient-kan/tree/master)

Learning Rate=10^-6:

<img width="600rem" alt="INTER lr-6 MLP_KAN" src="https://github.com/user-attachments/assets/8708cf06-d117-471a-9657-a5764010a419"/>

<a name="INTER2"/></a>

### KAN-based and non-KAN-based convolutional nets

Based on ![Convolutional-KANs by AntonioTepsich](https://github.com/AntonioTepsich/Convolutional-KANs)
and on ![KANvolver by Subhransu Sekhar Bhattacharjee ](https://github.com/1ssb/torchkan/tree/main)

<img width="600rem" alt="INTER lr-6 CONV" src="https://github.com/user-attachments/assets/c5332b11-f045-40cd-9aeb-83d2666d86b9"/>

<a name="EfficientKAN"></a>

## ❗ The Gaussian Peaks Problem
Here we show how the ![8th PyKAN regression example](https://github.com/KindXiaoming/pykan/blob/master/docs/Example/Example_8_continual_learning.ipynb) can be solved by EfficientKAN with the same performance as PyKAN.

Read more on ![_Something different from the official results for KAN_](https://github.com/Blealtan/efficient-kan/issues/38)

After introducing the *sb_trainable* and *sp_trainable* on the EfficientKAN class, and setting them to `False` just like PyKAN does, the same results can be achieved:

<img width="600rem" alt="Gaussian Peaks EfficientKAN" src="png/GaussianPeaksEfficientKAN.png"/>

<a name="Authors"></a>

## 👨🏻‍💻 Authors

| Name              | Email                       | GitHub                                          |
|-------------------|-----------------------------|-------------------------------------------------|
| Valerio Morelli   | s1118781@studenti.univpm.it | [MrPio](https://github.com/MrPio)               |
| Federica Paganica | s1116749@studenti.univpm.it | [Federica](https://github.com/federicapaganica) |
| Alessandro Cacciatore | a.cacciatore1@unimc.it | [geronimaw](https://github.com/geronimaw) |
