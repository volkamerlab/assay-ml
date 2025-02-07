# Research goals

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

## Assay split

### *Q1* How does aggregated data cluster wrt. assays?

The process of generating data leads to highly clustered datasets: A
lead compound is identified and data on slight variants is collected in
the iterative refinement and improvement of the lead compound. The data
deposited in public repositories such as ChEMBL often is collected this
way.

### *Q2* Do assay splits capture generalization performance better for real-world scenarios?

Recently, the splitting of molecular data has received a lot of
attention (DataSAIL, UMAP splits, people moving to scaffold splits).
Largely, these splitting schemes try to make the ML task "harder" by
testing the model on domain-shifted or even out-of-domain test data.
This, of course is somewhat contrary to the main principle of ML to
abstractly represent the training data domain.

We are interested in the predictive performance of a model on novel
compound classes and predictive quality for new experiments in labs that
did not generate the training data. Here, the nature of aggregated
public data sets offers a unique opportunity.

*Hypothesis*: Instead of splitting along scaffolds or ad-hoc embeddings
(FP UMAP), we can use the assays as a basis for our data split.

### *Q3* How do we evaluate models on assay splits?

Analysis of ChEMBL data shows how data naturally clusters along
different assays. For aggregating such data into large datasets for
machine learning this would not immediately pose problems if the
measurements were consistent. However, previous work (Landrum) has shown
that, especially for IC50 measurements, data between assays often is
inconsistent. This leads us to the following hypothesis.

*Hypothesis*: Models trained on aggregated and possibly inconsistent
data should not be evaluated in a way that assumes consistency. For
example, we cannot expect a model trained on IC50 values that is
inconsistent between assays to achieve good generalization on predicting
IC50 values. Instead, we should consider the ranking performance in
assays that were not considered in the training set.

### *Q4* Can we train models without aggregating different assays?

We established that aggregated data is not not suitable for training and
evaluating absolute predictions. Therefore it is natural to evaluate
models relatively (ie. ranking performance). Similarly, we can move from
absolute predictive models to relative predictive models. These are
models that rank sets of two or more molecules.

Such models could potentially be less influenced by the inter-assay
noise and therefore exhibit a superior relative predictive performance.
