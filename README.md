# A Weithing Loss Function for multi-class classification

The method described for this report aims to solve a multi-class classification problem, in particular,
to AIO defects. This method consists of three independent stages: pre-processing data,
classification, and last but not least post-processing data. 
The pre-possessing stage aims to enhance every image such that the most salient features are boosted. 
Indeed we show that using the pre-processing strategy increases the performance of our model by about 15%.
Our classifier is the convolution neural model (CNN) AlexNet.
We chose this architecture to prove that our strategy can increase the accuracy of a model,
without using more data nor deeper architectures as well.  
Lastly, the post-processing stage evaluates the mentioned classifier up to every class, .i.e., the probability estimation per class. 
Based on this class evaluation, a weighting vector is manually designed to redefine our cross-entropy function, 
then, we retrain our classifier again. Details of the proposed method are illustrated in bellow.
Note that in the post-processing stage, a repeating cycle is generated, where a model is trained, 
and then after to evaluate a weight vector, it is restrained again.
 
