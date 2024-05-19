# Multitask learning with BERT

This project focuses on developing a multi-task model for natural language processing related tasks. The tasks include sentiment analysis, paraphrase detection, and semantic textual similarity at the sentence-level. We implement a minBERT transformer-based model for this purpose and use that as our baseline model. Then, we use a combination of gradient surgery, regularization techniques such as the Smoothness-inducing Adversarial Regularization, and dropout layers to improve our model performance. We evaluate the impact of using these techniques on the model performance for the three listed tasks. On the test set, our best performing model achieves an accuracy of 72.6% for paraphrase detection, accuracy of 53.3% for sentiment analysis, and correlation of 0.427 for semantic textual similarity. The three-task average performance is 0.562 on the test set.

The poster (poster.jpg) provides a summary of this project.
Project Report (Project_report.pdf) provides a detailed explanation of the project.

<p align="center">
  <img src="poster.jpg" width="1000" title="hover text">
</p>
