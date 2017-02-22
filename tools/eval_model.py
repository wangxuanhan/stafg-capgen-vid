__author__ = 'Xuanhan Wang'
import cfg
from cfg import CFG
import data_reader
import numpy as np
import metric 

def evaluate(generator,keyword_generator=None,feat_type='ResNet',eval_type='valid'):
    if eval_type=='valid':
        num_test_samples = len(CFG['VALID'])
    elif eval_type=='test':
        num_test_samples = len(CFG['TEST'])
    print 'testing samples:%d'%(num_test_samples)
    resList= {}
    gtList = {}
    IDs = []
    test_id = 0
    print 'generating sentece...'
    for i in range(num_test_samples):
        if eval_type=='valid':
            _id = CFG['VALID'][i]
        elif eval_type=='test':
            _id = CFG['TEST'][i]
        vidID, capID = _id.split('_')
        if not resList.has_key(vidID):
            test_id += 1
            gtList[vidID] = data_reader.data_source['CAPs'][vidID]
            sent = sent_generate(generator, vidID,feat_type)
            resList[vidID] = [{u'image_id': vidID, u'caption': sent}]
            print '%d vidID: %s description: %s'%(test_id, vidID, sent)
            IDs.append(vidID)

    print 'evaluating...'
    scorer = metric.COCOScorer()
    eval_res = scorer.score(gtList, resList, IDs)
    return eval_res

def generate_keywords(generator,vidID,top_k=10,feat_type='ResNet'):
    feats = []
    if feat_type=='G':
        frame_feats = np.copy('/mnt/disk3/guozhao/features/MSR-VTT/googlenet/'+vidID+'.npy')
    elif feat_type=='InV3':
        frame_feats = np.load('/mnt/disk3/guozhao/features/MSR-VTT/inception-v3/'+vidID+'.npy')
    elif feat_type =='ResNet':
        frame_feats = np.load('/mnt/disk3/guozhao/features/MSR-VTT/ResNet_152/'+vidID+'.npy')
    elif feat_type=='c3d':
        frame_feats = np.load('/mnt/disk3/guozhao/features/MSR-VTT/C3D/'+vidID+'.npy')
    feats.append(data_reader.preprocess_vid_data(frame_feats))
    x_cnn = np.asarray(feats,dtype='float32')
    scores = generator(x_cnn)
    idx = np.argsort(scores.flatten())
    top_k_idx = idx[-top_k:]
    cap = ''
    for i in top_k_idx:
        cap = cap+CFG['idx2word'][i]+' '
    return cap,scores


def sent_generate(generator,vidID,feat_type='G',k=5):
    # beam search algorithm for sentence generation
    # change the path of features by yours
    feats = []
    c3d_feats = []
    if feat_type=='G':
        frame_feats = np.copy('/mnt/disk3/guozhao/features/MSR-VTT/googlenet/'+vidID+'.npy')
    elif feat_type=='InV3':
        frame_feats = np.load('/mnt/disk3/guozhao/features/MSR-VTT/inception-v3/'+vidID+'.npy')
    elif feat_type =='ResNet':
        frame_feats = np.load('/mnt/disk3/guozhao/features/MSR-VTT/ResNet_152/'+vidID+'.npy')
    elif feat_type=='c3d':
        frame_feats = np.load('/mnt/disk3/guozhao/features/MSR-VTT/C3D/'+vidID+'.npy')
    feats.append(data_reader.preprocess_vid_data(frame_feats))
    x_cnn_feats = np.asarray(feats,dtype='float32')
    x_cnn = np.copy(x_cnn_feats)
    # c3d_feats.append(data_reader.preprocess_vid_data(clip_feats,'C3D'))
    # x_3dcnn_feats = np.asarray(c3d_feats,dtype='float32')
    # x_3dcnn = np.copy(x_3dcnn_feats)
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
        p_words = generator(x_cnn, input_sents, input_masks)
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
            # if reach the end of sentence. record this sentence and scores
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
            # if it reaches the maxlength but generation of some sentences is not done yet
            if dead_k < k:
                top_k_sents += next_input_sents
                top_k_scores += cur_scores
            break
        x_sentence, gt, mask = data_reader.rnn_input_reorganized(next_input_sents)
        x_cnn = np.repeat(x_cnn_feats,x_sentence.shape[0],axis=0)
        # x_3dcnn = np.repeat(x_3dcnn_feats,x_sentence.shape[0],axis=0)
        prev_scores = np.asarray(cur_scores).reshape((-1,1))

    top_k_scores = np.asarray(top_k_scores)
    best_one = np.argsort(top_k_scores)[0]
    best_sent = top_k_sents[best_one]
    cap = [CFG['idx2word'][id] for id in best_sent]
    return ' '.join(cap)

def print_batch_sentence(sents,scores):
    for i in range(scores.shape[0]):
	cap = [CFG['idx2word'][id] for id in sents[i]]
	print 'cap:',cap,'scores:',scores[i]



def sent_generate_v1(generator, vidID, feat_type='ResNet'):
    # simple version of sentence generation 
    feats = []
    if feat_type=='G':
    	frame_feats = np.copy(data_reader.data_source['FEATs'][vidID])
    elif feat_type=='InV3':
	frame_feats = np.load('/home/guozhao/features/youtube/inception-v3/'+vidID+'.npy')
    elif feat_type =='ResNet':
	frame_feats = np.load('/mnt/disk2/wangxuanhan/datasets/VIDCAP/MSVD/RESNET101/'+vidID+'.npy')
    elif feat_type=='c3d':
        frame_feats = np.load('/mnt/disk2/wangxuanhan/datasets/VIDCAP/MSVD/C3D/'+vidID+'.npy')
    feats.append(data_reader.simple_comp_vid_level_feats(frame_feats))
    x_cnn = np.asarray(feats,dtype='float32')

    x_sentence = np.zeros((1, CFG['SEQUENCE LENGTH'] - 1), dtype='int32')
    mask = np.zeros((1,CFG['SEQUENCE LENGTH']),dtype='uint8')
    words = []
    i = 0
    while True:
        mask[0,i] = 1
        p0 = generator(x_cnn, x_sentence, mask)
        pa = p0.argmax(-1)
        tok = pa[0][i]
        word = CFG['idx2word'][tok]
        if word == '<eos>' or i >= CFG['SEQUENCE LENGTH'] - 1:
            return ' '.join(words)
        else:
            x_sentence[0][i] = tok
            words.append(word)
        i += 1

