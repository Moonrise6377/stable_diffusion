# Readme
## Content
- [Task](task1)
- [Description of the approach](task2)
- [Data](task3)
- [Results](task4)
- [What could be improved](task5)

## Task <a class="anchor" id="task1"></a>
As part of a course on NLP site ODS.ai , it is necessary to prepare the final text processing project. It is necessary to prepare a report on the example provided in the course, describe your approach, 
how the data was collected, describe the approach used. 
[Link to the course](https://ods.ai/tracks/nlp-course-spring-23 )

## Description of the approach <a class="anchor" id="task2"></a>
The pre-trained BLIP (Bootstrapped Language-Imaged Pre-trained) model was used as the main model. The main difference of the model is that the model uses
the CapFilt image detector to describe the image. It is expensive to prepare datasets with image descriptions entirely with the help of human labor. That's why CapFilt can help us here.
To reduce the cost of data collection, data is collected from the Internet, image-text pairs are collected using parsing. CapFilt works as follows: the model generates a description of the image,
and then compares the generated description with what was collected from the Internet and discards the least relevant. This allows you to clear the data from noise. In fact, we use another
neural network to clear the data from noise.  
In addition to using the pre-trained BLIP model, we also used other models to compare results and predictive ability. BLIP, ResNet & LSTM, CapDec, ClipCapm, VLKD were used.

### Metric
The BLEU metric was used as metrics.

## Data <a class="anchor" id="task3"></a>
In the project, we use the COCO dataset for 2014. The dataset is available for download on the official website. The COCO (Common Objects in Context) dataset is a widely used set
of reference data for object detection, segmentation, and subtitle creation tasks. We will focus on the part of this dataset containing image captions. The dataset contains five
man-made captions for each image in the train, val, and test sets.

## Results <a class="anchor" id="task4"></a>
The results of all models are presented below:
| Model         | BLEU-score |
|---------------|------------|
| BLIP          | 0.36       |
| ResNet & LSTM | 0.21       |
| CapDec        | 0.26       |
| ClipCap       | 0.32       |
| VLKD          | 0.17       |

## What could be improved <a class="anchor" id="task5"></a>
The model was not configured for the dataset. We can achieve better results and improve the metric if we try to select the parameters of the model. There is also already a BLIP-2 model, you can
try to apply it and look at the actual differences in the predictive ability of models between versions.
