# Introduction
Automatic video caption generation with STA-FG framework<br>
This code is used for the implementation of models which described in the paper titled [Fused-GRU for with Semantic-Temporal Attention for Video Captioning]()
# Requirements software
1. **Python 2.7**<br>The simplest way to install it is to use `Annaconda`.<br>
2. **Theano** <br> We recomend you to use the newest version. Simply type 
`pip install --upgrade https://github.com/Theano/Theano/archive/master.zip` to install the newest version.<br>
3. **Lasagne** <br> Our implementation is based on the lasagne. Type `pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip` to install the newest version of Lasagne.
4. **[coco-caption](https://github.com/tylin/coco-caption)**<br> Dowload this evaluation code. You need to add the path into your #PYTHONPATH or move it to the ${ROOTPATH}. 

# Preparation
To run this code, please follow intructions below.<br>
1. You need to create a dataset folder and dowload preproccessed dataset [here](). Then unzip the msvd.zip file which in the "data" folder into the dataset folder.<br>
2. Revise config file: tools/cfg.py and set your dataset path. 
3. Revise data processing file: tools/data_reader.py and reset your features path. 
# Training & Testing
1. run `scripts/script_train.py` to train a model and test it.
2. run `scripts/script_eval_stafg_rc.py` to evaluate models under the mutiple features setting.
