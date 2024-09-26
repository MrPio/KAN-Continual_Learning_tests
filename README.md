# Continual learning in KANs

This repository contains the codes for the paper [A preliminary study on continual learning in computer vision using Kolmogorov-Arnold Networks](https://arxiv.org/abs/2409.13550). We investigate the ability of Kolmogorov-Arnold Networks (KANs) to deal with computer vision tasks in a class-incremental learning scenario.

KAN were presented by Liu and colleagues in their work ![KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756).

---

### 📙 The slideshow presented on the day of the exam:

[KANs Continual Learning [Slideshow PPTX] - Morelli Valerio Paganica Federica.pptx](https://github.com/user-attachments/files/15993699/KANs.Continual.Learning.Morelli.Valerio.Paganica.Federica.pptx)

[KANs Continual Learning [Slideshow PDF] - Morelli Valerio Paganica Federica.pdf](https://github.com/user-attachments/files/15993700/KANs.Continual.Learning.Morelli.Valerio.Paganica.Federica.pdf)

---

<a name="index"></a>

## 📘 Table of Contents

* [⬇️ Class-IL Scenario](#CLASS)
  * [MLPs vs KANs](#CLASS1)
  * [KAN-based and non-KAN-based convolutional nets](#CLASS2)
* [❗ The Gaussian Peaks Problem](#EfficientKAN)
* [👨🏻‍💻 Authors](#Authors)

<a name="CLASS"/></a>

## ⬇️ Class-IL Scenario

<a name="CLASS1"/></a>
### MLP vs PyKAN vs EffKAN
🎬 The following videos highlight the difference between MLP, PyKAN (![PyKAN](https://github.com/KindXiaoming/pykan)), and EffKAN (![EfficientKAN](https://github.com/Blealtan/efficient-kan)) in a Class-IL scenario on the MNIST dataset. Each video shows the per-epoch predicitons of the corresponding model in the optimal hyper-parameter configuration.


<img width="600rem" alt="confusion_matrix_MLP" src="https://github.com/user-attachments/assets/3640dd72-7d0c-4599-bfa1-624ad82578c8"/>
<img width="600rem" alt="confusion_matrix_EffKAN" src="https://github.com/user-attachments/assets/d0f9bb21-5f83-45ae-8cc5-e5a91a01d086"/>
<img width="600rem" alt="confusion_matrix_KAN" src="https://github.com/user-attachments/assets/1a6337c2-c324-4ac6-9fc0-3de7dbb40670"/>

The following test accuracy plots show the same trainin runs as the confusion matrices.
<img width="600rem" alt="fig4plot_PyKAN" src="https://github.com/user-attachments/assets/07314cb4-751b-4a83-8548-504b9e32bd7d"/>
<img width="600rem" alt="fig4plot_MLP" src="https://github.com/user-attachments/assets/edc8787c-04aa-4ac7-ba1e-9371a5013a4d"/>
<img width="600rem" alt="fig4plot_EffKAN" src="https://github.com/user-attachments/assets/869828c7-2d5e-466d-916c-d712a582e3ef"/>


<a name="CLASS2"/></a>
### KAN-based and non-KAN-based convolutional nets
Based on ![Convolutional-KANs by AntonioTepsich](https://github.com/AntonioTepsich/Convolutional-KANs).

<img width="600rem" alt="ConvNetslr-6" src="[https://github.com/user-attachments/assets/c5332b11-f045-40cd-9aeb-83d2666d86b9](https://github.com/user-attachments/assets/b88e1dcf-baef-493e-af46-fc13c2c181bc)"/>

<a name="EfficientKAN"></a>

## ❗ The Gaussian Peaks Problem
Here we show how the ![8th PyKAN regression example](https://github.com/KindXiaoming/pykan/blob/master/docs/Example/Example_8_continual_learning.ipynb) can be solved by EfficientKAN with the same performance as PyKAN.

Read more on ![_Something different from the official results for KAN_](https://github.com/Blealtan/efficient-kan/issues/38)

After introducing the *sb_trainable* and *sp_trainable* on the EfficientKAN class, and setting them to `False` just like PyKAN does, the same results can be achieved:

<img width="600rem" alt="GaussianPeaksEfficientKAN" src="https://github.com/user-attachments/assets/9475487e-6ce7-448e-8ac5-2713f604b390"/>
<img width="600rem" alt="GaussianPeaksPyKAN" src="https://github.com/user-attachments/assets/b32523e7-c5bf-4829-a9da-7b2585d9a747"/>

<a name="Authors"></a>

## 👨🏻‍💻 Authors

| Name              | Email                       | GitHub                                          |
|-------------------|-----------------------------|-------------------------------------------------|
| Valerio Morelli   | s1118781@studenti.univpm.it | [MrPio](https://github.com/MrPio)               |
| Federica Paganica | s1116749@studenti.univpm.it | [Federica](https://github.com/federicapaganica) |
| Alessandro Cacciatore | a.cacciatore1@unimc.it | [geronimaw](https://github.com/geronimaw) |
