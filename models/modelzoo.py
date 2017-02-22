__author__ = 'Xuanhan Wang'
import sys
import numpy as np
sys.path.insert(0,'../tools')
import lasagne
from lasagne.layers import InputLayer, EmbeddingLayer,DenseLayer,GRULayer,\
        ConcatLayer, LSTMLayer, DropoutLayer, ReshapeLayer, FeaturePoolLayer
from lasagne.nonlinearities import softmax,linear,sigmoid
from CTXAttentionGRU import CTXAttentionGRULayer
import cfg
from cfg import CFG
import collections
from theano import tensor as T
def show_network_configuration(net):

    num_layers = net.__len__()
    layer_names = net.keys()
    print 40*'-'
    print 'network configuration'

    for i in range(num_layers):
        print 'Layer: %s shape:%r'%(layer_names[i],net[layer_names[i]].output_shape)


def build_STA_FG_Network():
    net = collections.OrderedDict()
    # INPUTS----------------------------------------
    net['sent_input'] = InputLayer((None, CFG['SEQUENCE LENGTH']),
                                   input_var=T.imatrix())
    net['word_emb'] = EmbeddingLayer(net['sent_input'], input_size=CFG['VOCAB SIZE']+3,\
                                    output_size=CFG['WORD VECTOR SIZE'],W=np.copy(CFG['wemb']))

    net['vis_input'] = InputLayer((None,CFG['VISUAL LENGTH'], CFG['VIS SIZE']))
    # key words model-------------------------------------
    net['vis_mean_pool'] = FeaturePoolLayer(net['vis_input'],
                                                CFG['VISUAL LENGTH'],pool_function=T.mean)
    net['ctx_vis_reshp'] = ReshapeLayer(net['vis_mean_pool'],(-1,CFG['VIS SIZE']))
    net['global_vis'] = DenseLayer(net['ctx_vis_reshp'],num_units=CFG['EMBEDDING SIZE'],nonlinearity=linear)
    net['key_words_prob'] = DenseLayer(DropoutLayer(net['global_vis']), num_units=CFG['VOCAB SIZE']+3,nonlinearity=sigmoid)
    # gru model--------------------------------------
    net['mask_input'] = InputLayer((None, CFG['SEQUENCE LENGTH']))
    net['s_gru'] = GRULayer(net['word_emb'],num_units=CFG['EMBEDDING SIZE'], \
                            mask_input=net['mask_input'],hid_init=net['global_vis'])
    net['sta_gru'] = CTXAttentionGRULayer([net['s_gru'],net['vis_input'],net['global_vis']],
                                           num_units=CFG['EMBEDDING SIZE'],
                                           mask_input=net['mask_input'])
    net['fusion'] = DropoutLayer(ConcatLayer([net['sta_gru'],net['s_gru']],axis=2), p=0.5)
    net['fusion_reshp'] = ReshapeLayer(net['fusion'], (-1,CFG['EMBEDDING SIZE']*2))
    net['word_prob'] = DenseLayer(net['fusion_reshp'], num_units=CFG['VOCAB SIZE']+3,
                                  nonlinearity=softmax)
    net['sent_prob'] = ReshapeLayer(net['word_prob'],(-1,CFG['SEQUENCE LENGTH'], CFG['VOCAB SIZE']+3))
    return net


def build_basic_model():
    net = collections.OrderedDict()
    net['sent_input'] = InputLayer((None, CFG['SEQUENCE LENGTH']),
                                   input_var=T.imatrix())
    net['word_emb'] = EmbeddingLayer(net['sent_input'], input_size=CFG['VOCAB SIZE']+3,\
                                    output_size=CFG['EMBEDDING SIZE'])
    net['vis_input'] = InputLayer((None, CFG['VIS SIZE']),
                                  input_var=T.matrix())
    net['vis_emb'] = DenseLayer(net['vis_input'], num_units=CFG['EMBEDDING SIZE'],
                                nonlinearity=lasagne.nonlinearities.identity)
    net['vis_emb_reshp'] = ReshapeLayer(net['vis_emb'],(-1,CFG['SEQUENCE LENGTH'],CFG['EMBEDDING SIZE']))
    net['decorder_input'] = ConcatLayer([net['vis_emb_reshp'], net['word_emb']],axis=2)
    net['feat_dropout'] = DropoutLayer(net['decorder_input'],p=0.5)

    net['mask_input'] = InputLayer((None, CFG['SEQUENCE LENGTH']))
    net['lstm1_1'] = LSTMLayer(net['feat_dropout'],num_units=CFG['EMBEDDING SIZE'], \
                            mask_input=net['mask_input'], grad_clipping=5.)
    net['lstm_dropout'] = DropoutLayer(net['lstm1_1'], p=0.5)
    net['lstm_reshp'] = ReshapeLayer(net['lstm_dropout'], (-1,CFG['EMBEDDING SIZE']))
    net['word_prob'] = DenseLayer(net['lstm_reshp'], num_units=CFG['VOCAB SIZE']+3,
                                  nonlinearity=softmax)
    net['sent_prob'] = ReshapeLayer(net['word_prob'],(-1,CFG['SEQUENCE LENGTH'], CFG['VOCAB SIZE']+3))
    return net

