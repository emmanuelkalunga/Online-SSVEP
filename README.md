# Online SSVEP-based BCI using Riemannian Geometry

## Description
An analysis of Riemannian geometry based methods for classfication in SSVEP-based BCI.
The algorithms are tested on data available at https://github.com/sylvchev/dataset-ssvep-exoskeleton

## Dependencies
* Matlab 7 or later versions
* Biosig toolbox: http://biosig.sourceforge.net/

## Data

The code is tested on data available [here](https://github.com/sylvchev/dataset-ssvep-exoskeleton/"data").
For for a quick run of the code, the data should be placed in the [/data](/data/) folder

## Main files

1. `plots.m`
	plot all figures
2. `tables.m`
	Draw main results tables
3. `ClassProb_3class.m` & `ClassProb_4class.m`
	Online evaluation of class probabilities probability threshold used in online algorithm.
	For 3 classes and 4 classes (SSVEP classes + resting class) respectively
4. `offline_basic_potato_3class.m`
	An offline analysis of the MDRD with and without riemannian potato applied for outliers removal.
	Classification on epoch taken from cue-onset t0.
	Only SSVEP classes are being used
5. `offline_opt_potato_3class`
	Similar to offline_basic_potato_3class.m, but epochs are taken from t0+2 sec
6. `online_cum_3class.m` & `online_cum_4class.m`
	Implementation of the online algorithm not including the curve criterion.
	The classifier output is the class whose probability is beyond the probability threshold.
	For 3c lasses and 4 classes (SSVEP classes + resting class) respectively
7. `online_curve_3class.m` & `online_curve_4class.m`
	Implementation of the full online algorithm
	For 3c lasses and 4 classes (SSVEP classes + resting class) respectively
8. `online_curve_potato_3class.m` & `online_curve_potato_4class.m`
	Implementation of the full online algorithm.
	Training data filtered with Riemannian potato form ouliers removal.
	For 3c lasses and 4 classes (SSVEP classes + resting class) respectively.
9. `online_curve_tLen_3class.m` & `online_curve_tLen_4class.m`
	Evaluation of the window size, a hyper-parameter in the online algorith.
	For 3c lasses and 4 classes (SSVEP classes + resting class) respectively
10.`riemannian_classification_path.m`
	produces the path taken by covariance matrices during experiment and how they are being classified.

### References

* E. Kalunga, S. Chevallier, and Q. Barthelemy, Research Report: Using Riemannian geometry for SSVEP-based Brain Computer Interface, http://arxiv.org/pdf/1501.03227.pdf
* A. Barachant, S. Bonnet, M. Congedo, C. Jutten, Multiclass brain-computer interface classication by Riemannian geometry, TBME, 2010, 2927-2935


