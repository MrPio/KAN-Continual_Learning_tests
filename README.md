# Continual learning in KANs
This repository contains the results of the tests carried out to prove the ability of the Kolmogorov-Arnold neural network to resist the catastrophic forgetting that heavily affects MLPs.

---

## 📘 The results of these tests are presented in detail in this paper: 

[KANs Continual Learning [V2] - Morelli Valerio Federica Paganica.pdf](https://github.com/user-attachments/files/15993786/Kolmogorov_Arnold_Networks.pdf)

## 📙 The slideshow presented on the day of the exam:

[KANs Continual Learning [Slideshow PPTX] - Morelli Valerio Paganica Federica.pptx](https://github.com/user-attachments/files/15993699/KANs.Continual.Learning.Morelli.Valerio.Paganica.Federica.pptx)

[KANs Continual Learning [Slideshow PDF] - Morelli Valerio Paganica Federica.pdf](https://github.com/user-attachments/files/15993700/KANs.Continual.Learning.Morelli.Valerio.Paganica.Federica.pdf)

---

# Testing different learning rate scales on MLPs and KANs at MNIST

![MLP_KAN_different_lrs](https://github.com/MrPio/KAN-Continual_Learning_tests/assets/22773005/f375374c-5890-4053-96bf-95ace0dda9bc)



# Sorted MNIST training set (INTRA training set sorting)
The first test tries to understand the impact of a non-shuffled trainset on the training of the different architectures.
This scenario is common in real-time applications where the order of the input data cannot be decided and the network may not see the sample of a particular class again.
This is a deliberate attempt to make the network's training less effective because the order of the training sets, as is well known in machine learning, should always be random in machine learning.
In these unfavourable conditions, the KANs prove to outperform the MLPs

The training set is sorted as follows:
![intra](https://github.com/MrPio/KAN-Continual_Learning_tests/assets/22773005/c7730268-f646-45b9-8587-0e4d742168db)

When MLPs see a new, previously unseen digit at the same learning rate, they tend to become distorted more quickly:
![INTRA dataset lr-5 ep1-2](https://github.com/MrPio/KAN-Continual_Learning_tests/assets/22773005/904e3324-47e5-4289-a8a8-694246622f03)

The results of this test are:

## MLPs vs KANs

![INTRAdataset_NON-CONV](https://github.com/MrPio/KAN-Continual_Learning_tests/assets/22773005/0dbc7894-a8da-4cd3-9362-1ba48a767974)

## KAN-based and non-KAN-based convolutional nets


![INTRAdataset_CONV2](https://github.com/MrPio/KAN-Continual_Learning_tests/assets/22773005/0b731558-8ffe-4576-9f42-f24896eabbec)


# Domain IL Scenario (INTER training set sorting)

🎬 The following video highlights the difference between MLPs and KANs in a Domain-IL scenario:

https://github.com/MrPio/KAN_tests/assets/22773005/b244367a-9af1-4b56-b005-bda6b788d810

![confusion_matrix](https://github.com/MrPio/KAN-Continual_Learning_tests/assets/22773005/4e0561e9-32b8-44d3-ab01-8fd197451940)

## MLPs vs KANs


Based on ![Convolutional-KANs by Blealtan](https://github.com/Blealtan/efficient-kan/tree/master)

Learning Rate=10^-6:


![INTER lr-6 MLP_KAN](https://github.com/MrPio/KAN_tests/assets/22773005/91bb539d-3355-451a-bc21-89e79e4af524)

## KAN-based and non-KAN-based convolutional nets
Based on ![Convolutional-KANs by AntonioTepsich](https://github.com/AntonioTepsich/Convolutional-KANs)
and on ![KANvolver by Subhransu Sekhar Bhattacharjee ](https://github.com/1ssb/torchkan/tree/main)
![INTER lr-6 CONV](https://github.com/MrPio/KAN-Continual_Learning_tests/assets/22773005/f100156a-93ba-40db-834a-28ccdf4a3903)
