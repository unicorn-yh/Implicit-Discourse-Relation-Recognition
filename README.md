# Implicit-Discourse-Relation-Recognition

<br>

## Abstract
Implicit discourse relationship identification aims to discover the semantic relationship between two sentences with missing discourse connectives. The discourse relationship between two text fragments plays a key role in many natural language processing tasks. Conjunctives strongly indicate the meaning of discourse relations, while discourses without conjunctives are called implicit discourse. Compared with explicit relations, implicit relations are harder to detect, and the key to their prediction is to correctly model the semantics of two discourse arguments, as well as the contextual interaction between them. This paper proposes several frameworks based on the BERT encoder model, including BERT base, ALBERT, DistilBERT base, RoBERTa and other pre-trained models.

<br>

## Background
A discourse usually contains one or more sentences to describe daily events and for people to exchange ideas and opinions. Since a chapter is usually composed of multiple text fragments, the relationship between text fragments should be considered in order to accurately understand the theme of the chapter. There are often connectives used to express relationships in the text, but there are no connectives between some text fragments, but there is an implicit relationship between them. The task of implicit textual relation recognition is to detect implicit relations and classify their meanings between two text fragments without connectives. The IDRR task has played a crucial role in various downstream natural language processing tasks such as text summarization, machine translation, etc. This paper provides solutions to various BERT pre-trained models for the implicit discourse relationship recognition task.

<br>

## Problem Description
Discourse Relation Recognition (DRR) aims to identify whether there is a logical relationship between two text fragments in a discourse; if it exists, it further classifies the relationship meaning into some predefined types, such as temporal (temporal), causal ( Contingency), comparison (comparison) and extension (expansion) and other relations. An article may contain one or more connective words between text fragments. A connective word is a vocabulary that can directly express the meaning of a certain relationship. For example, the connective word "because" usually expresses a causal relationship between two text fragments. If there are connectives in the discourse relation recognition, it is called Explicit Discourse Relation Recognition (EDRR), otherwise it is called Implicit Discourse Relation Recognition (IDRR). This experiment uses the Penn Discourse Tree Bank 2.0 (PDTB-2.0) corpus as the input sample of the model.

<br>

## Solution
Neural network experiments successively adopted different types of BERT models to train the PDTB-2.0 dataset and classify the implicit discourse relations of the chapters. Firstly, BERT BASE, ALBERT BASE, DistilBERT BASE and other models were used for training in the experiment, but the effect was not good, and the F1 values were 56.31, 56.81, and 50.19 respectively. After improvement, RoBERTa BASE, a model based on tag-dependent perceptual sequence generation, was used to achieve an F1 value of 62.08; and then tried to use the RoBERTa LARGE model, and through parameter adjustment, an F1 value of 69.76 was achieved. Experiments have verified that BERT can improve the accuracy and F1 value of many natural language processing and language modeling tasks, and it plays an important role in IDRR tasks.

<br>

## Experimental setup
Environment: Visual Studio Code 1.74.1 + Python 3.7.7 + NVDIA GeForce RTX 3090
Data: Penn Discourse Tree Bank 2.0（PDTB-2.0）

<br>

## Algorithm flowchart

![flowchart](README/flowchart.png)

<br>

## Result

| Model           | Results                                                      |
| :---------------: | :------------------------------------------------------------: |
| BERT BASE       | ![bertbase-output](README/bertbase-output.png) |
| ALBERT BASE     | ![albertbase-output](README/albertbase-output.png) |
| DistilBERT BASE | ![distilbert-output](README/distilbert-output.png) |
| RoBERTa BASE    | ![robertabase-output](README/robertabase-output.png) |
| RoBERTa LARGE   | ![roberta-large](README/roberta-large.png) |


<br>

As shown in table above, it can be observed that the effect of using DistilBERT BASE at the beginning is not good, only reaching the F1 value of 50.19. This is due to the streamlined architecture of the DistilBERT model, which performs knowledge distillation in the pre-training phase, resulting in a 40% reduction in model size compared to the BERT model, a reduction in the number of layers, and much fewer trainable parameters. Among them, BERT's word segmentation embedding and pooling layers were deleted at the same time, and finally the number of DistilBERT BASE layers was reduced by 2 times. After trying to train the model on BERT BASE and ALBERT BASE, the experimental effect has been slightly improved, that is, the F1 values of 56.31 and 56.81 were respectively achieved, and the training time is equivalent to twice that of the DistilBERT BASE model. 

In the experiment, the RoBERTa model was used again for IDRR task training. The BASE version achieved a higher F1 value of 62.08, while the performance improvement of the LARGE version was more significant, reaching an F1 value of 69.76. RoBERTa works much better than other BERT-like models because the model uses dynamic masking, where different parts of the sentence are masked at each iteration, making the model more performant; other BERT-like models use static masking. The batch size used by RoBERTa is much larger than that of BERT, and the training data set is also much larger than that of BERT, which is about 160 GB; while BERT only trains about 16 GB of data, so RoBERTa is more robust on IDRR tasks than Other BERT-like models are high.

<br>

## Conclusion
From this experiment, I learned how to use various BERT-like models to train the Implicit Discourse Relation Discrimination (IDRR) task, and gained a deep understanding of variables such as model framework, number of parameters, model size, hyperparameters, etc. influence on the training effect. Compared with DistilBERT BASE, ALBERT BASE, BERT BASE, and RoBERTa BASE, RoBERTa LARGE model training has reached the highest F1 value, improving the overall model performance and accuracy.
