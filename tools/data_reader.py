__author__ = 'Xuanhan Wang'
import cfg
from cfg import CFG
import numpy as np
import random
data_source = {}
data_source['CAPs'] = np.load(CFG['DATASET PATH'] + 'CAP.pkl')
data_source['KEYWORDs'] = np.load(CFG['DATASET PATH'] + 'MSRVTT_COARSE_CAP.pkl')

def get_words(vidID, capID,sent_type='original'):
        if sent_type=='original':
            caps = data_source['CAPs'][vidID]
        elif sent_type=='coarse':
            caps = data_source['KEYWORDs'][vidID]
        rval = None
        for cap in caps:
            if not cap.has_key('cap_id'):
                continue
            if cap['cap_id'] == capID:
                rval = cap['tokenized'].split(' ')
                break
        return rval


def simple_comp_vid_level_feats(frame_feats,pool_function=np.mean,axis=0):
    if frame_feats.shape[0]>CFG['VISUAL LENGTH']:
        idx = random.sample(range(frame_feats.shape[0]),CFG['VISUAL LENGTH'])
        idx.sort()
        sampled_frame_feats = frame_feats[idx]
    else:
        gap = CFG['VISUAL LENGTH'] - frame_feats.shape[0]
        sampled_frame_feats = np.concatenate([frame_feats,np.zeros((gap,frame_feats.shape[1]),dtype='float32')])

    return pool_function(sampled_frame_feats,axis=axis)

def preprocess_vid_data(frame_feats,feat_type='ResNet'):
    if feat_type=='ResNet':
        vis_len = CFG['VISUAL LENGTH']
    elif feat_type=='c3d':
        vis_len = 12
    if frame_feats.shape[0]>vis_len:
        idx = random.sample(range(frame_feats.shape[0]),vis_len)
        idx.sort()
        sampled_frame_feats = frame_feats[idx]
    else:
        gap = vis_len - frame_feats.shape[0]
        sampled_frame_feats = np.concatenate([frame_feats,np.zeros((gap,frame_feats.shape[1]),dtype='float32')])

    return sampled_frame_feats

def rnn_input_reorganized(sents):

    masks = []
    input_seqs = []
    predict_words = []
    for s in sents:
        mask = np.zeros((CFG['SEQUENCE LENGTH'],),dtype='uint8')
        mask[0] = 1  # <bos> must be inputted
        input_words = np.zeros((CFG['SEQUENCE LENGTH'],),dtype='int32')
        input_words[0] = len(CFG['idx2word']) - 1
        output_words = np.zeros((CFG['SEQUENCE LENGTH'],),dtype='int32')

        if len(s) < CFG['SEQUENCE LENGTH']-1:
            input_words[1:1+len(s)] = s[:]
            mask[1:1+len(s)] = 1
            output_words[:len(s)] = s[:]
            # if len(s) < CFG['SEQUENCE LENGTH']-2:
            #     output_words[len(s)+1] = 0
        else:
            input_words[1:] = s[:CFG['SEQUENCE LENGTH']-1]
            mask[:] = 1

        masks.append(mask)
        input_seqs.append(input_words)
        predict_words.append(output_words)

    return np.asarray(input_seqs,dtype='int32'), np.asarray(predict_words,dtype='int32'),\
           np.asarray(masks,dtype='uint8')

def get_attr_labels(sents):
    attrs = []
    for s in sents:
	mul_labels = np.zeros((CFG['VOCAB SIZE']+3,),dtype='int32')
	mul_labels[s] = 1
	mul_labels[2] = 0
	attrs.append(mul_labels)
    return np.asarray(attrs)

def get_batch_data(batch_idx, feat_type='G'):
    feats = []
    words = []
    keywords=[]
    c3d_feats = []
    for item in batch_idx:
        vidID, capID = item.split('_')
	if feat_type=='G':
            frame_feats = np.copy(data_source['FEATs'][vidID])
	elif feat_type=='InV3':
	    frame_feats = np.load('/home/guozhao/features/youtube/inception-v3/'+vidID+'.npy')
	elif feat_type=='ResNet':
#	    frame_feats = np.load('/mnt/disk2/wangxuanhan/DATASETS/VIDCAP/MSVD/RESNET/'+vidID+'.npy')
	    frame_feats = np.load('/mnt/disk3/guozhao/features/MSR-VTT/ResNet_152/'+vidID+'.npy')
	elif feat_type=='c3d':
	    # frame_feats = np.load('/mnt/disk2/wangxuanhan/datasets/VIDCAP/MSVD/C3D/'+vidID+'.npy')
	    frame_feats = np.load('/mnt/disk3/guozhao/features/MSR-VTT/C3D/'+vidID+'.npy')
        sentence = get_words(vidID, int(capID))
        coarse_sent = get_words(vidID,capID,'coarse')
        if coarse_sent is None:
            continue
        feats.append(preprocess_vid_data(frame_feats,feat_type))
	# c3d_feats.append(simple_comp_vid_level_feats(clip_feats,'C3D'))
        words.append([CFG['worddict'][w]
                if CFG['worddict'].has_key(w) and CFG['worddict'][w] < CFG['VOCAB SIZE'] else 1 for w in sentence])
        kw = []
        for w in coarse_sent:
            if CFG['worddict'].has_key(w) and CFG['worddict'][w] < CFG['VOCAB SIZE']:
                kw.append(CFG['worddict'][w])
        keywords.append(kw)
    k_words = get_attr_labels(keywords)
    words, out_words, masks = rnn_input_reorganized(words)
    return np.asarray(feats,dtype='float32'), words, out_words, masks,k_words

# def test_fn():
#     batch_idx = CFG['TRAIN'][:CFG['BATCH_SIZE']]
#     feats,words,masks = get_batch_data(batch_idx)
#     print feats.shape
#     print words.shape
#     print masks.shape

# if __name__ == '__main__':
#     test_fn()
