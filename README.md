# Logic Tensor Network for Concept Learning - Semantic Scene Image Intepretation (LTN-CL)
 The LTN-CL is an extension of the Logic Tensor Network (Serafini and Garcez, 2016), as implemented with Pytorch in the repository 
*[LTN-Pytorch](https://github.com/benediktwagner/LTN_Pytorch_New)*

# How to use
1. Clone the git repository using the command:
```
git clone https://github.com/SofoklisPK/LTN_CL.git
```
2. Inside the source code directory download the CLEVR dataset from *[CLEVR website](https://cs.stanford.edu/people/jcjohns/clevr/)* and unzip.


The direcotry should look as such:
```
-LTN_CL\
--Experiment_1_version\
--Experiment_2_version\
--Experiment_3_version\
--images\
--scenes_train.json
--scenes_val.json
-- ...rest of source code files
```

To train a model run, either from main directory or the experiment subdirectories, depending on what version of the LTN-CL you would like to run:
```
python ltn_cl.py
```

To test the trained model:
```
python test_ltn_cl.py
```

# Experiment versions
There exist three different versions of the LTN-CL available to train and run in this repository. Each version corresponds to an experiment conducted for the purposes of a MSc Thesis (report to be made available soon).
