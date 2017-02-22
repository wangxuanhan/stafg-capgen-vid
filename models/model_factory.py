__author__ = 'Xuanhan Wang'
from modelzoo import build_basic_model,build_STA_FG_Network
import theano
from theano import tensor as T
import lasagne
import sys
sys.path.insert(0,'../tools')
from cfg import CFG
import collections

def show_network_configuration(net):

    num_layers = net.__len__()
    layer_names = net.keys()
    print 40*'-'
    print 'network configuration'

    for i in range(num_layers):
        print 'Layer: %s shape:%r'%(layer_names[i],net[layer_names[i]].output_shape)

def multi_label_loss(prediction, gt):
    # simple version of multi-label loss
    corrects = T.log(prediction)
    corrects = corrects*gt
    logits = T.sum(-corrects,axis=1)
    return logits

def classifier_train_test_config(net,model):
    net_params = lasagne.layers.get_all_params(net['word_prob'],trainable=True)
    mask = lasagne.layers.get_output(net['mask_input'])
    targets = T.ivector()
    # coherence sentence loss
    train_prob = lasagne.layers.get_output(net['word_prob'])
    train_class_loss = lasagne.objectives.categorical_crossentropy(train_prob,targets)
    train_class_loss = T.reshape(train_class_loss,(-1,mask.shape[1])) * mask
    train_class_loss = T.sum(train_class_loss.dimshuffle(0,1,'x'),axis=1)
    train_loss = T.mean(train_class_loss)
    train_pred = T.argmax(train_prob,axis=1)
    train_acc = T.mean(T.eq(train_pred,targets))
    # attention regularization terms
    sent_prob = lasagne.layers.get_output(net['sent_prob'],deterministic=True)
    att_out = net['sta_gru'].get_output_attention()
    # To avoid zero values which will cause the problem of Nan, we need to smooth attention values.
    att_out = att_out * mask[:,:,None] + CFG['epsilon']
    c_att_loss = T.sum(T.sum(att_out,axis=1)**2,axis=1)**(0.5)
    c_att_loss = T.mean(c_att_loss)
    s_att_loss = T.sum(T.sum(att_out**0.5,axis=2)**2,axis=1)
    s_att_loss = T.mean(s_att_loss)
    att_loss = c_att_loss+s_att_loss*0.01
    # semantic loss
    key_word_targets = T.imatrix()
    key_words_prob = lasagne.layers.get_output(net['key_words_prob'])
    semantic_loss = multi_label_loss(key_words_prob,key_word_targets)
    semantic_loss = CFG['beta']*T.mean(semantic_loss)
    # params regularization
    reg_cost = lasagne.regularization.regularize_network_params(net['word_prob'],lasagne.regularization.l2)

    model['semantic loss'] = semantic_loss
    model['key words'] = key_word_targets
    model['keyword prob'] = lasagne.layers.get_output(net['key_words_prob'],deterministic=True)
    model['attention loss'] = att_loss
    model['ground truth'] = targets
    model['train loss'] = train_loss + att_loss + semantic_loss + CFG['lambda']*reg_cost
    model['train accuracy'] = train_acc
    model['sent prob'] = sent_prob
    model['params'] = net_params[1:]
    model['attention'] = att_out
    return model

def train_test_func(MODEL,optimizer=1,learning_rate = 0.005):
    exp_funcs = {}
    if optimizer == 0:
        print 'Optimizer: general sgd '
        updates = lasagne.updates.sgd(MODEL['train loss'], MODEL['params'][:], learning_rate)
    else:
        print 'Optimizer: adadelta'
        updates = lasagne.updates.adadelta(MODEL['train loss'], MODEL['params'][:], learning_rate)

    print 'General settings: Learning rate(%f)  Momentum(%f)  NumberFunctions(%d) NumberParams(%d)'%\
          (learning_rate,0.9,2,MODEL['params'].__len__())
    print MODEL['params']
    updates = lasagne.updates.apply_momentum(updates,MODEL['params'][:])

    net_name = MODEL['net name']
    exp_funcs['train func'] = theano.function(inputs=[MODEL[net_name]['vis_input'].input_var,
#						    MODEL[net_name]['c3d_input'].input_var,
                                                    MODEL[net_name]['sent_input'].input_var,
                                                    MODEL[net_name]['mask_input'].input_var,
                                                    MODEL['ground truth'],MODEL['key words']],
                               outputs=[MODEL['train loss'],MODEL['train accuracy'],MODEL['attention'],MODEL['attention loss'],MODEL['semantic loss']],
                               updates=updates)
    exp_funcs['sent prob']= theano.function(inputs=[MODEL[net_name]['vis_input'].input_var,
#						    MODEL[net_name]['c3d_input'].input_var,
                                                    MODEL[net_name]['sent_input'].input_var,
                                                    MODEL[net_name]['mask_input'].input_var],
                              outputs=MODEL['sent prob'])
    exp_funcs['key word'] = theano.function(inputs=[MODEL[net_name]['vis_input'].input_var],
                                            outputs=MODEL['keyword prob'])
    return exp_funcs

def build_vidcaption_model(net_name='STA_FG'):
    net = build_STA_FG_Network()
    model = collections.OrderedDict()
    model['net name'] = net_name
    model[net_name] = net
    model = classifier_train_test_config(net,model)
    return model
