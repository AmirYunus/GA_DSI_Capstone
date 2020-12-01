
<h1>DETECTING ECONOMIC CRIME USING DEEP AUTOENCODER NEURAL NETWORK<span class="tocSkip"></span></h1>

Author: Amir Yunus<br>
GitHub: https://github.com/AmirYunus/GA_DSI_Capstone
***

# PREFACE

## Abstract

Learning to detect economic crime in large-scale accounting data is one of the long-standing challenges in financial statement audits or fraud investigations. Currently, the majority of applied techniques refer to hand-crafted rules derived from known fraud scenarios. Whilst fairly successful, these rules exhibit the drawback that they often fail to generalise beyond known fraud scenarios and fraudsters gradually find ways to circumvent them.

To overcome this disadvantage, we propose the application of a deep autoencoder neural network to detect anomalous journal entries. We demonstrate that the trained network’s reconstruction error obtainable for a journal entry can be interpreted as a highly adaptive anomaly assessment.

Experiments on three datasets of journal entries, show the effectiveness of the approach resulting in high F$_1$-Scores of 0.971 (train dataset) and 1.000 (test dataset). Our experiment also resulted in less false positive alerts compared to baseline methods. Initial feedback received by peers underpinned the quality of the approach in capturing highly relevant accounting anomalies.

## Index Terms

Accounting Information Systems · Enterprise Resource Planning (ERP) · Computer Assisted Audit Techniques (CAATs) · Journal Entry Testing · Forensic Accounting · Fraud Detection · Forensic Data Analytics · Deep Learning

## Motivation

The Association of Certified Fraud Examiners (ACFE) estimates that a typical organisation loses 5% of its annual revenues due to fraud. Economic crime, or commonly known as fraud, refers to "the abuse of one's occupation for personal enrichment through the deliberate misuse of an organisation's resources or assets".

> **" The median loss of a single financial statement fraud case is USD 150 thousand. The duration from the fraud perpetration till its detection was 18 months. "** - Association of Certified Fraud Examiners

A similar study, conducted by PricewaterhouseCoopers (PwC), revealed that nearly 25% of respondents experienced losses between USD 100 thousand and USD 1 million due to fraud. The study also showed that financial statement fraud caused by far the highest median loss of the surveyed fraud schemes. At the same time, organisations accelerate the digitisation and reconfiguration of business processes affecting in particular the Accounting Information Systems (AIS) or commonly known as Enterprise Resource Planning (ERP) systems. SAP, one of the most common ERP providers, estimates that approximately 77% of the world's transaction touches one of their systems.

> **" 49 percent of respondents said that their organisation has been a victim of fraud or economic crime in the past 24 months. "** - PricewaterhouseCoopers

Generally, international audit standards require the direct assessment of journal entries to detect potentially fraudulent activities. These techniques, usually based on some known fraudulent scenarios, are often referred to as "red-flag" tests or statistical analyses such as Benford's Law. However, these tests fail to generalise beyond historical fraud cases and therefore, unable to detect contemporary fraud methods.

![Relationship of Actor and Victimised Organisation](/images/04.png) 
<div align="right">- PricewaterhouseCoopers</div>

Recent developments in deep learning enables data scientists to extract complex, non-linear features, from raw sensory data, leading to advancements across many domains such as computer vision and speech recognition. This method can supplement the accountants and forensic examiners toolbox.

![Fraction of Internal Actors Conducting Economic Crime](/images/05.png) 
<div align="right">- PricewaterhouseCoopers</div>

In order to conduct fraud, perpetrators need to deviate from regular system usage or posting pattern. Such deviations are recorded by a very limited number of "anomalous" journal entries. Our anomaly assessment is highly adaptive and it allows to flag entries as "anomalous" if they exceed a predefined scoring threshold. We evaluate the proposed method based on three anonymized datasets of journal entries extracted from large-scale SAP ERP systems.

> **" Our ERP applications touch 77% of global transaction revenue. "** - SAP

In section 2, we provide an overview of related works in the field of fraud detection. Section 3 follows with a description of the autoencoder network architecture and presents the proposed methodology to detect accounting anomalies. The experimental setup and results are outlined in section 4 and section 5.

# Related Work

The task of detecting fraud and accounting anomalies has been studied by both practitioners and academia. Several references describe different fraud schemes and ways to detect unusual and "creative" accounting practices.

## Fraud Detection in Enterprise Resource Planning Data

Bay, et al. used Naive Bayes methods to identify suspicious general ledger accounts, by evaluating attributes derived from journal entries measuring any unusual general ledger account activity. Their approach was enhanced by McGlohon, et al. by applying link analysis to identify groups of high-risk general ledger accounts.

Kahn, et al. created transaction profiles of SAP ERP users. Similarly, Islam, et al. used SAP R/3 system audit logs to detect known fraud scenarios and collusion fraud via a "red-flag" based matching of fraud scenarios.

Debreceny and Gray analysed dollar amounts of journal entries obtained from 29 US organisations. In their work, they searched for violations of Benford’s Law, anomalous digit combinations as well as unusual temporal pattern such as end-of-year postings. More recently, Poh-Sun, et al. demonstrated the generalisation of the approach by applying it to journal entries obtained from 12 non-US organisations. 

Jans, et al. used latent class clustering to conduct a univariate and multivariate clustering of SAP ERP purchase order transactions. The approach was enhanced by a means of process mining to detect deviating process flows in an organisation procure to pay process. Transactions significantly deviating from the cluster centroids are flagged as "anomalous" and are proposed for a detailed review by auditors.

Argyrou et al. evaluated self-organising maps to identify "suspicious" journal entries of a shipping company. In their work, they calculated the Euclidean distance of a journal entry and the code-vector of a self-organising map's best matching unit. In subsequent work, they estimated optimal sampling thresholds of journal entry attributes derived from extreme value theory.

Concluding from the reviewed literature, the majority of references draw either on:

>* Historical accounting and forensic knowledge about various "red-flags" and fraud schemes; or
>* Traditional non-deep learning techniques

As a result, we see a demand for unsupervised and novel approaches capable to detect so far unknown scenarios of fraudulent journal entries.

## Anomaly Detection using Autoencoder Neural Networks

Currently, autoencoder networks have been widely used in image classification, machine translation and speech processing. Hawkins, et al. and Williams, et al. were probably the first who proposed autoencoder networks for anomaly detection. 

Since then, the ability of autoencoder networks to detect anomalous records was demonstrated in different domains such as X-ray images of freight containers, the KDD99, MNIST, CIFAR-10 as well as several other datasets from the UCI Machine Learning Repository. Zhou and Paffenroth enhanced the standard autoencoder architecture by an additional filter layer and regularisation penalty to detect anomalies.

Cozzolino and Verdoliva used the autoencoder reconstruction error to detect pixel manipulations of images. The method was enhanced by recurrent neural networks to detect forged video sequences. Paula, et al. used autoencoder networks in export controls to detect traces of money laundering and fraud by analysing volumes of exported goods.

# Detection of Accounting Anomalies

In this section, we introduce the main elements of autoencoder neural networks. We furthermore describe how the reconstruction error of such networks can be used to detect anomalous journal entries in large-scale accounting data.

## Deep Autoencoder Neural Networks

An autoencoder or replicator neural network defines a special type of feed-forward multilayer neural network that can be trained to reconstruct its input. The difference between the original input and its reconstruction is referred to as reconstruction error.

![Autoencoder Architecture](/images/06.png) 
<div align="right">- Marco Schreyer and Timur Sattarov</div>

In general, autoencoder networks are comprised of two non-linear mappings referred to as encoder and decoder network. Most commonly, the encoder and the decoder are of symmetrical architecture consisting of several layers of neurons each followed by a non-linear function and shared parameters. The encoder maps an input to a compressed representation in the latent space. This latent representation is then mapped back by the decoder to a reconstructed vector of the original input space. 

The autoencoder is then trained to learn a set of optimal encoder-decoder model parameters that minimises the dissimilarity of a given journal entry and its reconstruction as faithfully as possible. For binary encoded attribute values, as used in this work, the model measures the deviation between two independent multivariate Bernoulli distributions.

To prevent the autoencoder from learning the identity function, the number of neurons of the networks hidden layers are reduced using a "bottleneck" architecture. Imposing such a constraint onto the network’s hidden layer forces the autoencoder to learn an optimal set of parameters that result in a "compressed" model of the most prevalent journal entry attribute value distributions and their dependencies.

## Classification of Accounting Anomalies

We assume that the majority of journal entries recorded relates to regular business activities. In order to conduct fraud, perpetrators need to deviate from the "normal". Such behaviour will be recorded by a very limited number of journal entries. Breunig, et al. distinguished two classes of anomalous journal entries, namely global and local anomalies.

![Global and Local Anomalies](/images/07.png) 
<div align="right">- Marco Schreyer and Timur Sattarov</div>

Generally, "red-flag" tests performed by auditors are designed to capture such anomalies. However, such tests often result in a high volume of false positive alerts due to events such as reverse postings, provisions and year-end adjustments usually associated with a low fraud risk. This type of anomaly is significantly more difficult to detect since perpetrators intend to disguise their activities by imitating a regular activity pattern. As a result, such anomalies usually pose a high fraud risk since they correspond to processes and activities that might not be conducted in compliance with organisational standards.

In this work, we detect any unusual value or unusual combination of values observed as anomalies. This was proposed by Das and Schneider on the detection of anomalous records in categorical datasets.

## Scoring of Accounting Anomalies

Our score accounts for any "unusual" attribute value occurring in the journal entry. We do this by taking the reconstructed vectors from the decoder, and perform a matrix subtraction on the original input vectors. Next, we take the mean values of each journal entry and this gives us the reconstruction error.

Given that anomalous entries tend to be 0.02% of all entries recorded for the year, the model will sort the reconstruction error in descending order and focuses on the top entries. Next, a threshold score of 0.019 is applied and entries with reconstruction errors above the threshold will be flagged as anomalous.

# Experimental Results

This section describes the results of our evaluation. Upon successful training we evaluated the proposed scoring according to two criteria:
>* Are the trained autoencoder architectures capable of learning a model of the regular journal entries and thereby detect the anomalies?
>* Are the detected anomalies ”suspicious” enough to be followed up by accountants or forensic examiners?

## Quantitative Evaluation

To quantitatively evaluate the effectiveness of the proposed approach, a range of evaluation metrics including recall, precision and f1-Score. The choice of f1-Score is to account for the highly unbalanced anomalous vs. non-anomalous class distribution of the datasets.

|                   	| Train<br>Dataset 	| Test<br>Dataset 	| Clean<br>Dataset 	|
|-------------------	|:----------------:	|:---------------:	|:----------------:	|
|       Recall      	|       1.00       	|       1.00      	|       1.00       	|
|     Precision     	|       0.943      	|       1.00      	|        N/A       	|
|      f1-Score     	|       0.971      	|       1.00      	|        N/A       	|

As seen from the table above, the metrics are high for all datasets. Note that the clean dataset has no anomalies and therefore, unable to provide a precision and f1-Score.

Overall, the deep autoencoder neural network model satisfies our quantitative evaluation.

## Qualitative Evaluation

To qualitatively evaluate the character of the detected anomalies we need to review all journal entries detected by the architecture.

The majority of anomalies probably correspond to journal entries that exhibit one or two rare attribute values. This could be:

>* posting errors due to wrongly used general ledger accounts;
>* journal entries of unusual document types containing extremely infrequent tax codes;
>* incomplete journal entries exhibiting missing currency information;
>* shipments to customers that are invoiced in different than the usual currency;
>* products send to a regular client but were booked to another company code;
>* postings that exhibit an unusual large time lag between document date and posting date;
>* irregular rental payments that slightly deviate from ordinary payments.

All of the above indicated a weak control environment around certain business processes of the investigated organization. As the dataset is anonymised as received we are unable to ascertain for a fact whether these entries were truly anomalous, other than the label provided. It requires domain knowledge and the actual dataset, both of which the author does not have.

## Baseline Evaluation

There are various baseline models for us to consider, such as PCA, kMeans, One-Class Support Vector Machine, Local-Outlier Factor and DBSCAN. However, for simplicity of this report, we will only consider kMeans and Local-Outlier Factor. For purposes of evaluation, our metrics will be on recall, accuracy and f1-score.

|                   	| Autoencoder 	|  kMeans 	| Local<br>Outlier Factor 	|
|-------------------	|:-----------:	|:-------:	|:-----------------------:	|
|  True<br>Positive 	|     100     	|    0    	|            15           	|
| False<br>Negative 	|      0      	|   100   	|            85           	|
| False<br>Positive 	|      6      	| 330,243 	|            92           	|
|  True<br>Negative 	|   532,903   	| 202,666 	|         532,817         	|
|       Recall      	|    1.000    	|  0.000  	|          0.150          	|
|     Precision     	|    0.943    	|  0.000  	|          0.140          	|
|      f1-Score     	|    0.971    	|  0.000  	|          0.145          	|

As seen above, our autoencoder architecture is performing well above the baseline models.

# Conclusion and Future Work

In this work we presented a deep learning based approach for the detection of anomalous journal entries in large scaled accounting data. Our empirical evaluation demonstrates that the reconstruction error of deep autoencoder networks can be used as a highly adaptive anomaly assessment of journal entries. In our experiments we achieved a superior f1-Score of 0.971 in the train dataset compared to regular machine learning methods.

In this architecture, we looked only at the reconstruction errors. We could also perform clustering at the latent space and detect anomalous entries. This would be an area for future development. A python script is also made available to run the autoencoder architecture on your own dataset. A graphical user interface (GUI) would be something for the author to consider in the near future.

Given the tremendous amount of journal entries recorded by organisations annually, an automated and high precision detection of accounting anomalies can save auditors considerable time and decrease the risk of fraudulent financial statements.

## Appendix

The code we used to train and evaluate our models is available at:
https://github.com/AmirYunus/GA_DSI_Capstone

## Acknowledgement

This report is heavily influenced by Marco Schreyer and Timur Sattarov in their paper "Detection of Anomalies in Large-Scale Accounting Data using Deep Autoencoder Networks". They can be reached at marco.schreyer@dfki.de and sattarov.timur@pwc.com.

The author of this report is indebted to their work in advancements in detecting accounting anomalies. The author gives his thanks and any credit should be due to them.
