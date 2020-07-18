# Hcpcs2Vec

This project uses the skip-gram model to learn meaningful, dense embeddings for HCPCS procedure codes using big Medicare data.

## Inspiration

HCPCS procedure codes are used by medical professionals, billing companies, and insurance companies to document specific services provided to patients.

Procedure codes have proven to be valuable for a variety of analytics and machine learning projects, e.g. predicting readmission rates, predicting insurance fraud, or summarizing medical documents.

Representing these procedures using a traditional one-hot encoding results in very high-dimensional sparse vectors, i.e. > 50K dimensions. These large representations increase the computational complexity of machine learning algorithms and add to the curse of dimensionality. Furthermore, the one-hot representation fails to encode the meaningful relationships that exist between various procedure codes. Instead, all pairs of procedure codes are equidistant.

We can leverage the advances in NLP and word embeddings to construct low-rank continuous reprsentations of HCPCS codes that capture semantic meaning.

## Skip-Gram

Similar to NLP word embeddings with skip-gram, this project uses skip-gram to predict context HCPCS procedures that co-occur with a target HCPCS code.

The skip-gram model is a neural network (NN) whose architecture is defined by the cardinality of the input (vocab size) and the dimensionality of the desired embedding. The target word is provided to the NN as a one-hot vector and a softmax output layer is used to predict the probability of observing each word in the vocab. Once training has converged, the hidden layer is used to extract the learned embeddings.

To overcome the complexity of a very large softmax output layer, negative sampling only updates the weights for a sample of the non-contextual (negative) procedure codes.

![Skip-Gram Model Outline](assets/model-outline.png)

The model is fed two HCPCS procedure codes at a time with a label of 1 if they exist in context and 0 if they do not. Their embeddings are looked up, and their dot product is taken to compute their similarity. Finally, the similarity is fed to a binary logit output to predict whether or not they exist in context.

## Medicare Data

The Medicare data used in this project can be downloaded from the [cms.gov download page](https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/Physician-and-Other-Supplier).
