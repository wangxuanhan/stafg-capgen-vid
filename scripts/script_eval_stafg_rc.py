__author__ = 'Xuanhan Wang'
import sys
sys.path.insert(0,'../tools')
sys.path.insert(0,'../models')
import numpy as np
import cPickle
import lasagne
from lasagne.layers import InputLayer, EmbeddingLayer,DenseLayer,ElemwiseSumLayer,\
        ConcatLayer, LSTMLayer, DropoutLayer, ReshapeLayer,GRULayer,FeaturePoolLayer,\
        NonlinearityLayer
from lasagne.nonlinearities import softmax,linear,sigmoid
from CTXAttentionGRU import CTXAttentionGRULayer
import cfg
from cfg import CFG
import collections
import theano
from theano import tensor as T
import metric
import data_reader

def build_c3d_stafg():
	
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
    net['c3d_input'] = InputLayer((None,CFG['C3D LENGTH'], 4096))
    net['mask_input'] = InputLayer((None, CFG['SEQUENCE LENGTH']))
    net['sgru'] = GRULayer(net['word_emb'],num_units=CFG['EMBEDDING SIZE'], \
                            mask_input=net['mask_input'],hid_init=net['global_vis'])
    net['sta_gru'] = CTXAttentionGRULayer([net['sgru'],net['c3d_input'],net['global_vis']],
                                           num_units=CFG['EMBEDDING SIZE'],
                                           mask_input=net['mask_input'])
    net['fusion'] = DropoutLayer(ConcatLayer([net['sta_gru'],net['gru']],axis=2), p=0.5)
    net['fusion_reshp'] = ReshapeLayer(net['fusion'], (-1,CFG['EMBEDDING SIZE']*2))
    net['word_prob'] = DenseLayer(net['fusion_reshp'], num_units=CFG['VOCAB SIZE']+3,
                                  nonlinearity=softmax)
    net['sent_prob'] = ReshapeLayer(net['word_prob'],(-1,CFG['SEQUENCE LENGTH'], CFG['VOCAB SIZE']+3))
    return net

def build_res_stafg():

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
    net['sgru'] = GRULayer(net['word_emb'],num_units=CFG['EMBEDDING SIZE'], \
                            mask_input=net['mask_input'],hid_init=net['global_vis'])
    net['sta_gru'] = CTXAttentionGRULayer([net['sgru'],net['vis_input'],net['global_vis']],
                                           num_units=CFG['EMBEDDING SIZE'],
                                           mask_input=net['mask_input'])
    net['fusion'] = DropoutLayer(ConcatLayer([net['sta_gru'],net['gru']],axis=2), p=0.5)
    net['fusion_reshp'] = ReshapeLayer(net['fusion'], (-1,CFG['EMBEDDING SIZE']*2))
    net['word_prob'] = DenseLayer(net['fusion_reshp'], num_units=CFG['VOCAB SIZE']+3,
                                  nonlinearity=softmax)
    net['sent_prob'] = ReshapeLayer(net['word_prob'],(-1,CFG['SEQUENCE LENGTH'], CFG['VOCAB SIZE']+3))
    return net

def sent_func(net,feat_type='c3d'):
    exp_funcs = {}
    sent_prob = lasagne.layers.get_output(net['sent_prob'],deterministic=True)
    if feat_type == 'c3d':
        exp_funcs['sent prob']= theano.function(inputs=[net['vis_input'].input_var,
                                                    net['c3d_input'].input_var,
                                                    net['sent_input'].input_var,
                                                    net['mask_input'].input_var],
                              outputs=sent_prob)
    else:
        exp_funcs['sent prob']= theano.function(inputs=[net['vis_input'].input_var,
                                                    net['sent_input'].input_var,
                                                    net['mask_input'].input_var],
                              outputs=sent_prob)
    return exp_funcs

def sent_generate(c3d_generator,res_generator,vidID,feat_type='G',k=5):
    feats = []
    c3d_feats = []
    if feat_type=='G':
        frame_feats = np.copy(data_reader.data_source['FEATs'][vidID])
    elif feat_type=='InV3':
        frame_feats = np.load('/home/guozhao/features/youtube/inception-v3/'+vidID+'.npy')
    elif feat_type =='ResNet':
        frame_feats = np.load('/mnt/disk3/guozhao/features/MSR-VTT/ResNet_152/'+vidID+'.npy')
#    elif feat_type=='c3d':
    clip_feats = np.load('/mnt/disk3/guozhao/features/MSR-VTT/C3D/'+vidID+'.npy')
    feats.append(data_reader.simple_comp_vid_level_feats(frame_feats))
    x_cnn_feats = np.asarray(feats,dtype='float32')
    x_cnn = np.copy(x_cnn_feats)
    c3d_feats.append(data_reader.simple_comp_vid_level_feats(clip_feats,'C3D'))
    x_3dcnn_feats = np.asarray(c3d_feats,dtype='float32')
    x_3dcnn = np.copy(x_3dcnn_feats)
    # initial input
    x_sentence = [np.zeros((CFG['SEQUENCE LENGTH'],), dtype='int32')]
    x_sentence[0][0]=len(CFG['idx2word'])-1
    mask = [np.zeros((CFG['SEQUENCE LENGTH'],),dtype='uint8')]
    mask[0][0] = 1
    iword = 0
    dead_k = 0
    prev_scores = np.zeros((1))
    top_k_sents = []
    top_k_scores = []
    while True:
        input_sents = np.asarray(x_sentence)
        input_masks = np.asarray(mask)
        p_words = c3d_generator(x_cnn, x_3dcnn, input_sents, input_masks)+res_generator(x_cnn, input_sents, input_masks)
        p_words = p_words[:,iword,:]
        current_scores = prev_scores - np.log(p_words)
        cand_scores_flat = current_scores.flatten()
        p_top_k_idx = np.argsort(cand_scores_flat)[:k - dead_k]
        batch_idx = p_top_k_idx / p_words.shape[1]
        word_idx = p_top_k_idx % p_words.shape[1]
        cur_top_live_k_scores = cand_scores_flat[p_top_k_idx]

        next_input_sents = []
        cur_scores = []
        for idx,[batch_i, word_i] in enumerate(zip(batch_idx, word_idx)):
            c_word = CFG['idx2word'][word_i]
            
            if c_word == '<eos>':
                dead_k += 1
                if iword>0:
                   top_k_sents.append(x_sentence[batch_i][1:1+iword])
                top_k_scores.append(cur_top_live_k_scores[idx])
                continue
            # update prev_scores and model input for next step
            tmp_sent = []
            for i in range(iword+1):
                if i==0:
                   continue
                tmp_sent.append(input_sents[batch_i,i])
            tmp_sent.append(word_i)
            next_input_sents.append(tmp_sent)
            cur_scores.append(cur_top_live_k_scores[idx])
        iword += 1
        if dead_k == k or iword >= CFG['SEQUENCE LENGTH'] - 1:
            
            if dead_k < k:
                top_k_sents += next_input_sents
                top_k_scores += cur_scores
            break
        x_sentence, gt, mask = data_reader.rnn_input_reorganized(next_input_sents)
        x_cnn = np.repeat(x_cnn_feats,x_sentence.shape[0],axis=0)
        x_3dcnn = np.repeat(x_3dcnn_feats,x_sentence.shape[0],axis=0)
        prev_scores = np.asarray(cur_scores).reshape((-1,1))

    top_k_scores = np.asarray(top_k_scores)
    best_one = np.argsort(top_k_scores)[0]
    best_sent = top_k_sents[best_one]
    cap = [CFG['idx2word'][id] for id in best_sent]
    return ' '.join(cap)

def evaluate(c3d_generator,res_generator,feat_type='ResNet'):
    num_test_samples = len(CFG['TEST'])
    print 'testing samples:%d'%(num_test_samples)
    resList= {}
    gtList = {}
    IDs = []
    test_id = 0
    print 'generating sentece...'
    for i in range(num_test_samples):
        _id = CFG['TEST'][i]
        vidID, capID = _id.split('_')
        if not resList.has_key(vidID):
            test_id += 1
            gtList[vidID] = data_reader.data_source['CAPs'][vidID]
            sent = sent_generate(c3d_generator,res_generator, vidID,feat_type)
            resList[vidID] = [{u'image_id': vidID, u'caption': sent}]
            print '%d vidID: %s description: %s\t'%(test_id, vidID, sent)
            IDs.append(vidID)
    print 'evaluating...'
    scorer = metric.COCOScorer()
    eval_res = scorer.score(gtList, resList, IDs)
    return eval_res


if __name__=='__main__':
   c3d_f = open('../models/c3d_model_name','r')
   c3d_param = cPickle.load(c3d_f)
   c3d_f.close()
   res_f = open('../models/res_model_name','r')
   res_param = cPickle.load(res_f)
   res_f.close()
   print 'params ok'
   res_decoder = build_res_stafg()
   lasagne.layers.set_all_param_values(res_decoder['word_prob'],res_param)
   c3d_decoder = build_c3d_stafg()
   lasagne.layers.set_all_param_values(c3d_decoder['word_prob'],c3d_param)
   print 'nets ok'
   res_fn = sent_func(res_decoder,feat_type='res')
   c3d_fn = sent_func(c3d_decoder)
   print 'funcs ok'
   print 'evaluating...'
   eval_res = evaluate(c3d_fn['sent prob'],res_fn['sent prob'])
   for metric, score in eval_res.items():
       print '%s: %.3f'%(metric, score),
   print '\n'
   print 'DONE'
