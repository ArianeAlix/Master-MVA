# Course about Graphs in Machine Learning

## Homeworks done
- **TP1**: Spectral Clustering
- **TP2**: Semi-Supervised Learning (SSL), and application to face recognition
- **TP3**: Graph Neural Networks

<br/>

## Personal project on drug repurposing

### Title
Network-based approach for drug repurposing using drug signature and disease phenotype

### Abstract of the paper
Drug repurposing consists of the investigation of already used drugs, to see if they can be used for treating other diseases. This approach has recently raised thanks to the access to large-scale perturbation databases, such as the Library of Integrated Network-based Cellular Signatures (LINCS) and the development of new computational methods. In this paper, we review a drug–disease associations prediction method based on a recommendation system (bipartite graph) where the studied features are the drug signatures and diseases phenotypes.

### Conclusion

We implemented a new method of drug repurposing based on Alaimo’s recommendation method
(DT-Hybrid, see [14]). We used the Statistically Significant Connectivity Map metric (ssCMap) to
compute the similarities between drugs based on their gene signatures, and between diseases based
on their gene phenotypes. One could use other metrics, like XCos, to see if it could improve our
results.
By using cross-validation, we chose the best parameters to use in the computation of the method
according to their ability to "recover" deleted drug-disease associations.

#### Interpretation of the results
We must note that the result of 61.54% and 74.56% for the recovery metrics (see 4.1) is quite good
considering the form of the data. Indeed, it is important to note that the random partitioning method
associated with the cross-validation can cause the isolation of some nodes in the network on which
the tests are being performed. A main limitation of recommendation algorithms like in our method is
the inability to predict new interactions for drugs or diseases for which no information is available.
This implies that in the presence of isolated nodes a bias is introduced in the evaluation of results
[13].
Furthermore, we computed recommendations for all diseases and drugs even if we had no information
on them in the signature or association tables by filling the tables with 0. As a consequence, bad
final scores in the recommendation tables might be caused by a lack of information rather than
an incompatibility between the drug and disease. We must note that it is preferable, since drug
repurposing should rely on "worst-case" scenario in the face of uncertainty as a safety precaution.
Additionally, it would be interesting to check for all of the recommended drugs if they were reported
as effective for the corresponding diseases in the literature. Unfortunately, we did not have the names
associated to most of the diseases (only ids like ’c3665869 ’).

### References
[1] Subramanian et al. (November 2017). A Next Generation Connectivity Map: L1000 platform and
the first 1,000,000 profiles

[2] Q. Duan et al. (August 2016). L1000CDS2: LINCS L1000 characteristic direction signatures
search engine

[3] Clark NR, Hu KS, Feldmann AS et al. (2014). The characteristic direction: a geometrical
approach to identify differentially expressed genes.

[4] Musa et al. (2018). A review of connectivity map and computational approaches in pharmacogenomics

[5] Zhang SD, Gant T. A (2008). A simple and robust method for connecting small-molecule drugs
using gene-expression signatures.

[6] J. Cheng et al. (2013). Evaluation of analytical methods for connectivity map data.

[7] J. Cheng et al. (2014). Systematic evaluation of connectivity map for disease indications.

[8] J. Lamb et al. (2006). The connectivity map: using gene-expression signatures to connect small
molecules, genes, and disease.

[9] Qu XA, RajPal DK. (2012). Applications of connectivity map in drug discovery and development.

[10] F. Ioro et al. (2010). Discovery of drug mode of action and drug repositioning from transcriptional
responses.

[11] F. Ioro et al. (2013). Network based elucidation of drug response: from modulators to target.

[12] C. Pacini, F. Ioro et al. (2013). DvD: An R/Cytoscape pipeline for drug repurposing using public
repositories of gene expression data

[13] S. Alaimo, R. Giugno, A. Pulvirenti et al. (2016). Recommendation Techniques for Drug–Target
Interaction Prediction and Drug Repositioning.

[14] S. Alaimo, A. Pulvirenti, R. Giugno et al. (2013). Drug–target interaction prediction through
domain-tuned network-based inference.

[15] S. Alaimo, V. Bonnici, D. Cancemi et al. (2015). Dt-web: a web-based application for drugtarget
interaction and drug combination prediction through domain-tuned network-based inference.

[16] J. Moynihan and H. Moore (2010). Endocrine system dynamics and MS epidemiology.

[17] Müssig K, Gallwitz B et al. (2007). Pegvisomant treatment in gigantism caused by a growth
hormone-secreting giant pituitary adenoma.

[18] Naila Goldenberg, Michael S. Racine et al. (2008). Treatment of Pituitary Gigantism with the
Growth Hormone Receptor Antagonist Pegvisomant.

