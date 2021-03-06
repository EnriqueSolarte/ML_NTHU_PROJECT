# A Weithing Loss Function for multi-class classification

The method described for this project aims to solve a multi-class classification problem, in particular,
to AOI defects as described in [Aidea](https://aidea-web.tw/topic/a49e3f76-69c9-4a4a-bcfc-c882840b3f27), with limited dataset, and a simple CNN architecture. This method consists of three independent stages: pre-processing data,
classification, and last but not least post-processing data. 
The pre-possessing stage aims to enhance every image such that the most salient features are boosted. 
Indeed we show that using the pre-processing strategy increases the performance of our model by about 15%.
Our classifier is the convolution neural network (CNN) AlexNet.
We chose this architecture to prove that our strategy can increase the accuracy of a model,
without using more data nor deeper architectures as well.  
Lastly, the post-processing stage evaluates the mentioned classifier up to every class, .i.e., the probability estimation per class. 
Based on this class evaluation, a weighting vector is manually designed to redefine our cross-entropy function, 
then, we retrain our classifier again. Details of the proposed method are illustrated in bellow.
Note that in the post-processing stage, a repeating cycle is generated, where a model is trained, 
and then after to evaluate a weight vector, it is restrained again.

## Figure 1. Teaser. 
In this figure, the three main stages of our model are illustrated: pre-processing, classification, post-processing. Note that the post-processing stage defines multiple cycles, wherein our classifier is trained based on the preprocessed dataset, and then based on a hand-made weighting vector it is re-trained multiple times. A weighting vector w is defined at every cycle. These weitghting vectors are listed in [table 2](#table-1-weighting-vector---mask)


![Figure 1](figures/teaser.png)


## Figure 2. Training pipeline

This flowchart illustrates the training process, from model-0 until our final model-6. Note that in the training process different settings well as weighting vectors (masks) were used. For further details please refer to [table 1](#table-1-weighting-vector---mask) and [table 2](#table-2-hyperameter-for-training).


![Figure 2](figures/training.png)


## Table 1. Weighting Vector - masks

![Table 1](figures/masks.png)


## Table 2. Hyperameter for training

![Table 2](figures/hyperparameters.png)


## Table 3. Models' performance
 Note that the last column is the accuracy reached by the evaluation in the Aidea website. Using the same architecture and the same dataset we incrementally increase the accuracy of our model by using our proposed weighting loss function.

![Table 2](figures/model_performance.png)


## Figure 3. Comparison 1
Note that the purple evaluation describes the accuracy (a) and loss evaluation (b) for the model-0 without using the pre-processing. Moreover, the cyan evaluation shows how effective our weighting strategy is by further reducing the loss evaluation (b) and increasing the accuracy (a). The orange line represents the evaluation of the model-0 by using a pre-processing stage (Image enhancement) 

![Table 2](figures/comparison1.png)

## Figure 4. Comparison 2
Note that in panel (a) and (b) a comparison between model-1 and model-4 is presented.  Consider that the model-4 (red line) is re-trained from model-1 by using the mask-1 described in [Table 1](#table-1-weighting-vector---mask). In panel (c) a comparison between model-4, model-5, and model-6 are illustrated. Note that those latter were trained by using the mask-2 in [Table 1](#table-1-weighting-vector---mask). 


![Table 2](figures/comparison2.png)

## Implementation
The implementation for this project is described as follows:
```
root/
├── data/
|    ├── train_images
|    |    ├── ...*.png 
|    ├── test_images
|    |    ├── ...*.png
├── figures/
├── log/
├── src/
|    ├── classifier.py
|    ├── filters.py
|    ├── reading_data.py
|    └── setup.py
├── evaluate_classifier.py 
├── evaluate_test_data.py
└── training_classifier.py
```
In order to use this classifier the following scripts can be taken as reference ```evaluate_classifier.py```, ```evaluate_test_data.py```, and ```training_classifier.py```.

The usage for these scripts are described as follows:

**evaluate_classifier.py**
```py
from evaluate_classifier import eval_model

config = dict(shape=(224, 224),
             log="RUNNING", # Log directory where the model were stored
             pre_trained="08-12.57") # partial name of the trained model

eval_model(cfg=config)
```


**evaluate_test_data.py**
```py
from evaluate_test_data import eval_model

config = dict(shape=(224, 224),
            log="RUNNING", # Log directory where the model were stored
            pre_trained="08-12.57", # partial name of the trained model
            extra="final")
eval_model(cfg=config)
```



**training_classifier.py**
```py
from training_classifier import train

config = dict(shape=(224, 224),
                batch=200, # batch-size
                dip=True, # Whether pre-processing or not
                random=False, # Random pre-processing or not
                epochs=100, # Num. epochs
                lr=1e-3,  # learning rate
                dc_st=10, # Epochs for decay lr
                dc=0.9, # decay ratio 
                mt=0.9, # Momentum
                model="AlexNet", # Architecture model
                msk="model0", # used mask (see classifier.py)
                pre_trained=None,  # partial name of the pre-trained model
                extra="Model_DIP") # extra info 
train(cfg=config)
```


## Authors:

Shri Harish - 
mshriharish@gmail.com

Bolivar Enrique Solarte -
enrique.solarte.pardo@gmail.com 

