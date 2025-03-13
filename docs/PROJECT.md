# Assay-based machine learning

Machine learning has become a standard component in many drug design
pipelines. A central part in designing and evaluating such models is to
evaulate the generalizability of such models. Especially interesting is
the predictive accuracy on compounds that are not present and event very
different from the training data.

The urge to assess generalizability away from the training domain stems
from the practical scenarios where such models are used. In
(computer-aided) drug-design researchers typically aim to identify
compounds with certain properties (typically the inhibition of a known
target, no toxicological activity, solubility, etc.). The two main
scenarios here are lead identification and lead optimization.

In lead identification ML is usually applied in either a generatively or
in a virtual screening setting. Since here we are concerned with
molecular property prediction, we will focus on virtual screening. Here,
a large library of molecules is scanned using an ML model to predict
properties of interest. For this task the out-of-domain performance of
an ML model is crucial. Since these are the compounds researchers are
most interested in.

In lead optimization a compound with beneficial properties has been
identified, and researchers are interested in modifying that compound to
enhance certain properties. In this setting the out-of-domain
performance can be less important when employing techniques such as
fine-tuning.

## Research

### *Q1* How does aggregated data cluster wrt. assays?

The process of generating data leads to highly clustered datasets: A
lead compound is identified and data on slight variants is collected in
the iterative refinement and improvement of the lead compound. The data
deposited in public repositories such as ChEMBL often is collected this
way.

We can perform an experiment to figure out, how much information is
contained in assay assignments alone:

1.  Split the data via {scaffold split, UMAP split, ....}
2.  Use the training set assay mean as a predictor
3.  Evaluate test set performance

Performing this experiment on ChEMBL data already gives us a performance
that would be considered quite good in many settings!

*Hypothesis*: **Assays cluster in the molecular space and in terms of
measured activity.**


### *Q2* Do assay splits capture generalization performance better for real-world scenarios?

Recently, the splitting of molecular data has received a lot of
attention (DataSAIL, UMAP splits, people moving to scaffold splits).
Largely, these splitting schemes try to make the ML task "harder" by
testing the model on domain-shifted or even out-of-domain test data.
This, of course is somewhat contrary to the main principle of ML to
abstractly represent the training data domain.

In virtual screenings, we are interested in the predictive performance
of a model on novel compound classes and predictive quality for new
experiments in labs that did not generate the training data. Here, the
nature of aggregated public data sets offers a unique opportunity.

Instead of splitting along scaffolds or ad-hoc embeddings (FP UMAP), we
can use the assays as a basis for our data split. Then the evaluation on
an unseen assays would represent how valuable the model's prediction
would have been on that assay. To make this more realistic still, these
splits along assays could be temporal to reflect the actual knowledge in
the repository up to the study.

*Hypothesis*: **Assay splits are a closer representation of real-world
virtual screening scenarios.**

The arguments for this hypothesis are philosophical and coined more
towards virtual screening. The studies could be expanded to look at how
transfer learning using a few data points of unseen assays would impact
performance. For this question the ML methodology takes a center role
and that would dilute the purpose of this project a little.

Another important point is the experimental bias due to different
experimental settings: Different labs have slightly different working
procedures; they use different assay providers; and more. This could
introduce shifts in the reported activity values. This kind of noise
naturally is highly problematic if our goal is to predict absolute
values such as an IC50 or a solubility. This leads us to our next
question:

### *Q3* How do we evaluate models on assay splits?

Analysis of ChEMBL data shows how data naturally clusters along
different assays. For aggregating such data into large datasets for
machine learning this would not immediately pose problems if the
measurements were consistent. However, previous work (Landrum) has shown
that, especially for IC50 measurements, data between assays often is
inconsistent. This leads us to the following hypothesis.

*Hypothesis*: **Models trained on aggregated and possibly inconsistent
data should not be evaluated in a way that assumes consistency.**

For example, we cannot expect a model trained on IC50 values that is
inconsistent between assays to achieve good generalization on predicting
IC50 values. Instead, we should consider the ranking performance in
assays that were not considered in the training set.

If we are forgoing the assumption of inter-assay consistency of
measurements, we are left with the relative performance of different
compounds. Essentially, this results in a censoring of the absolute
values and instead relying on pairwise differences. If the data quality
is especially suspect, we could censor the data even more and transform
data to a classification problem: "Is compound A *better* than compound
B?"

For evaluation this implies a shift of test objective. Now we either
have a classification target or the prediction of a delta between
compound values. The central change is, that these are only evaluated on
pairs that have been tested in a single assay.

Naturally, this change of objective can also be transferred to the model
training and design.

### *Q4* Can we train models without aggregating different assays?

We established that aggregated data is not not suitable for training and
evaluating absolute predictions. Therefore it is natural to evaluate
models relatively (ie. ranking performance). Similarly, we can move from
absolute predictive models to relative predictive models. These are
models that rank sets of two or more molecules.

Such models could potentially be less influenced by the inter-assay
noise and therefore exhibit a superior relative predictive performance.

We perform a study comparing absolute "inter-assay" prediction models
against ranking "intra-assay" models. For a comparison of approaches, we
choose a pairwise model for its relative simplicity.

The design space is infinite. It has the same number of parameters and
architecture as the model for absolute predictions. In the pairwise
model, an embedding of both inputs is computed using the same network.
Than the difference of the embeddings is passed through a readout
module. Architecturally, the this difference of embeddings is the only
change to the absolute prediction model.

*Hypothesis*: **Rank-based models can achieve a better performance in
ordering compounds in unseen assays.**

#### Modelling study

*Pair models*: Pair models predict the difference between two query
molecules instead of predicting values for molecules in isolation. This
shift to the domain of molecular tuples allows us to decide whether we
want to learn on inter-assay information or on intra-assay information
alone.

*Set rank models*: We focus on the SetRank model. This shifts the focus
to ranking arbitrarily large sets of molecules. Building on the assay
split idea, we can define these sets to be data points from single
assays. This way we circumvent learning from low confidence data points
which in this context are sets of measurments originating from separate
assays.

For both modelling paradigms, we can compare performance of the exact
same architecture when training on only within and inter assay data. For
datasets based on ChEMBL data, we observe a marked difference:
Intra-assay training out-performs all-data training. Overall ensemble
prediction (i.e. set rank models and pair prediction) out-perform
standard IC50 prediction.

There is a trade-off between coverage of the domain and the data
quality. When make the step to set models, we change the domain to the
comparison of molecules. Due to the clustering of molecular structures
along assays, the variety of learning only on these sets is usually less
than when taking arbitrary sets. Depending on consistency of
measurements between assays, we will either observe a benefit or not of
using only within-assay data.

### Further points

### Censored data and classification / ranking models

Sometimes data is censored. Consider IC50 data, as an example: Often
there is a maximal measured concentration. If there was no inhibition \>
50% at that maximal concentration, no IC50 can be reasonably determined.
Such information cannot really be handled in a principled way when doing
absolute IC50 prediction. If we consider the model as a better/worse
classification, we can still incorporate the available negative data.

If we do not want to go for classification alone, we can also build a
multi-objective model with a classification and a delta-regression
target.

## Conclusions

Aggregating data from different experiments can introduce noise. This
noise stems from inconsistencies between their measurements. The problem
is aggravated by the heavy clustering of compounds by experiment.
Furthermore there is little overlapping data to harmonize measurements.
Moving from absolute to relative predictions and splitting the data
along experiments could give a better impression of model performance in
realistic settings. These issues can also be addressed in model design
and training.
