# Project for the Deep Learning for Medical Imaging course

## Title
Vessel Extraction from Retinal Images (DRIVE dataset)

## Abstract
In this work, we present a description and evaluation of our
method for the segmentation of blood vessel of retinas. Retinal vessel
segmentation can be of great interest for the diagnosis of retinal vascular
diseases such as age-related macular degeneration, glaucoma and diabetic
retinopathy. Therefore, in recent years, several automatic segmentation
methods have been proposed, ranging from the use of simple filters to
Deep Neural Networks.
The methods based on Convolutional Neural Networks mainly rely on
the learning of local patterns of vessels, but do not consider the graphical
structure of those. We think that taking advantage of the vessels network-
like form, inherent to their biological utility, would help improve the vessel
segmentation accuracy. Therefore, our method uses both a IterNet [16]
architecture, intertwined with a Graph Neural Network with Attention to
simultaneously comprehend local, global and structural vessel patterns.
The idea of a combination of a Convolutional Neural Network and a
Graph Network comes from Shin's "VGN" [14], a method developed in
2018 that was the state-of-the-art on retinal vessels segmentation when it
got out.
We evaluated our model on a retinal image called DRIVE, and compared
it to the current state-of-the-art methods in terms of the average precision,
AUC and F-measure. The scores obtained on our validation set are on-par
with thee state-of-the-art.

### Results
|              | IterNet only | IterNet only | VGN with IterNet |  VGN with IterNet  |
|--------------|--------------|------------|------------------|---------------------------------|
|              | Training     | Validation | Training         | Validation                      |
| AUC          | 0.9810       | 0.9842     | 0.9826           | 0.9848 |
| F1-score     | 0.7914       | 0.7965     | 0.8062           | 0.7962                          |



### References
1. G. Azzopardi, N. Strisciuglio, M. Vento, and N. Petkov, \Trainable cosfire filters for
vessel delineation with application to retinal images," Medical image analysis, vol.
19, pp. 46{57, 1. (2015)
2. Soares, J.V.B., Leandro et al., "Retinal vessel segmentation using the 2-D gabor
wavelet and supervised classification." IEEE Trans. Med. Imaging 25 (9), 1214{1222.
doi: 10.1109/TMI.2006.879967. (2006)
3. Y. Zhao, Y. Liu et al., "Retinal vessel segmentation: An efficient graph cut ap-
proach with retinex and local phase." PloS One 10 (4), e0122332. doi: 10.1371/jour-
nal.pone.0127486. (2015)
4. S.Y. Shin , S. Lee et al. , "Extraction of coronary vessels in 
uoroscopic X-ray
sequences using vessel correspondence optimization." In: Ourselin, S., Joskowicz,
L., Sabuncu, M.R., Unal, G., Wells, W. (Eds.), Medical Image Computing and
Computer-Assisted Intervention (MICCAI). Springer International Publishing, Cham,
pp. 308{316 . (2016)
5. Straat, Michiel and Jorrit Oosterhof. "Segmentation of blood vessels in retinal fundus
images." ArXiv abs/1905.12596 (2019)
6. Becker et al., "Supervised feature learning for curvilinear structure segmentation."
In: Mori, K., Sakuma, I., Sato, Y., Barillot, C., Navab, N. (Eds.), Medical Image
Computing and Computer-Assisted Intervention (2013).
7. A. Sironi, V. Lepetit, P. Fua, "Projection onto the manifold of elongated structures for
accurate extraction." In: Proceedings of IEEE International Conference on Computer
Vision (ICCV), pp. 316{324. doi: 10.1109/ICCV.2015.44. (2015)
8. M. M. Fraz, P. Remagnino et al., "An ensemble classification-based approach applied
to retinal blood vessel segmentation." IEEE Transactions on Biomedical Engineering,
vol. 59, pp. 2538{2548. (2012)
9. J. I. Orlando, E. Prokofyeva and M. B. Blaschko, "A Discriminatively Trained
Fully Connected Conditional Random Field Model for Blood Vessel Segmentation in
Fundus Images," in IEEE Transactions on Biomedical Engineering, vol. 64, no. 1,
pp. 16-27, (Jan. 2017)
10. P. Liskowski and K. Krawiec, "Segmenting Retinal Blood Vessels With Deep Neural
Networks," in IEEE Transactions on Medical Imaging, vol. 35, no. 11, pp. 2369-2380,
(Nov. 2016)
11. \DRIVE Dataset." http://www.isi.uu.nl/Research/ Databases/DRIVE/, (2004)
12. R. Sharma , A. Mugeesh and K. Nama, "DRIVE - Digital Retinal Images for
Vessel Extraction", https://github.com/rohit9934/DRIVE-Digital-Retinal-Images-
for-Vessel-Extraction
13. Olaf Ronneberger, Philipp Fischer, Thomas Brox, "U-Net: Convolutional Networks
for Biomedical Image Segmentation", Computer Science Department and BIOSS
Centre for Biological Signalling Studies, University of Freiburg, Germany. (2015)
14. S. Y. Shin et al., "Deep vessel segmentation by learning graphical connectivity",
Medical Image Analysis 58 (2019)
