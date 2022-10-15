DeepGenGrep: a general deep learning-based predictor for multiple genomic signals and regions


INSTALLATION
To run DeepGenGrep:
First install scikit-learn (http://scikit-learn.org/), tensorflow-gpu (https://pypi.org/project/tensorflow-gpu/) 


EXAMPLE:
All datasets employed are included in this project. These datasets are collected from the following literature.

Kalkatawi, M., et al. DeepGSR: an optimized deep-learning structure for the recognition of genomic signals and regions. Bioinformatics 2019;35(7):1125-1132. 

-----------------------train model----------------------------------------------------------------------------------

Train DeepGenGrep on human genome for TIS recognition as follows:   

python DeepGenGrep.py -gsr TIS -org hs -len 600

Train DeepGenGrep on human genome for the recognition of PAS AATAAA signal as follows:

python DeepGenGrep.py -gsr PAS_AATAAA -org hs -len 600

Parameters and description are as follows:
--gsr: genome signal region (TIS, PAS, PAS_AATAAA, Splice_acc, Splice_don)
-org: organism name, hs (human), mm (mouse), bt (bovine), dm (Drosophila melanogaster)
-len: the length of input sequence

The input dataset was first randomly split into the benchmark (75%) and independent test (25%) datasets. Then, the benchmark dataset was further divided into the training and validation datasets with a ratio of 8:2.

Trained model is saved in the ‘Model’ fold, and the evaluated results is saved in ‘Results’ fold. Please do not change the directory including 'data', 'Model', and 'Results' in the project. 

---------------------Predciton-----------------------------------------------------------------------------------

When we train the model on TIS and PAS signals, each sequence consists of 300 upstream nucleotides, the considered signal variant and 300 downstream nucleotides. As a result, each PAS sequence has 606 nucleotides, while each TIS sequence has 603 nucleotides. For the splice site training dataset, the length of each sequence is 398-nt.
When we predict TIS and PAS signals in test sequences. The length of input sequences is also 603 and 606 for prediction of TIS and PAS, respectively. If the length of sequences is less than 603 (for TIS) and 606 (for PAS), these sequences would be filtered out. If the length of sequences is more than 603 (for TIS) and 606 (for PAS), the model would cut long sequences into multiply overlap sequences with 603bp (for TIS) or 606 bp (for PAS).

The common line of prediction is as follows:

python prediction.py -gsr -TIS -org hs, -input test.fasta -out out

----------------------------------------------------------------------------------------------------------------
if you can doanloed the data please use the Goole Drive.  [data]: https://drive.google.com/file/d/1P4MgOK7BVf9D1zbysPt3aHaK3IUhbG63/view?usp=sharing
----------------------------------------------------------------------------------------------------------------

CONTACTS


If you have any queries, please email to liuqzhong@nwsuafe.du.cn
