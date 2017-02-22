__author__ = 'Xuanhan Wang'
# model configures 
import numpy as np
CFG = {}
CFG['BATCH_SIZE'] = 256
CFG['NUM_EPOCH'] = 20
CFG['USE_MODEL'] = 0
CFG['MODEL PATH'] = '' # trained model path
CFG['SEQUENCE LENGTH'] = 39
CFG['VISUAL LENGTH'] = 35
CFG['C3D LENGTH'] = 12
CFG['EMBEDDING SIZE'] = 512
CFG['WORD VECTOR SIZE'] = 300
CFG['VOCAB SIZE'] = 21585
CFG['VIS SIZE'] = 2048
CFG['DATASET PATH'] = ''
CFG['TRAIN'] = np.load(CFG['DATASET PATH']+'train.pkl')
CFG['TEST'] = np.load(CFG['DATASET PATH'] +'test.pkl')
CFG['VALID'] = np.load(CFG['DATASET PATH'] + 'valid.pkl')
msr_corpus = np.load(CFG['DATASET PATH'] + 'msrvtt_corpus.pkl')
CFG['worddict'] = msr_corpus['dict'] 
CFG['wemb'] = np.asarray(msr_corpus['emb'],dtype='float32')
CFG['lambda'] = 1e-6
CFG['beta'] = 0.1
CFG['epsilon'] = 0.000001
# wordict start with index 2
word_idict = dict()
for kk, vv in CFG['worddict'].iteritems():
    word_idict[vv] = kk
word_idict[0] = '<eos>'
word_idict[1] = 'UNK'
word_idict[len(word_idict)] = '<bos>'
CFG['idx2word'] = word_idict
