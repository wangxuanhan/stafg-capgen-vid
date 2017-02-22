__author__ = 'Xuanhan Wang'
import _init_paths
import model_factory
import eval_model
import data_reader as data
from cfg import CFG
import numpy as np
import cPickle
import time
import lasagne
#-------------prepare models---------------------------
def exp_model():
    if CFG['USE_MODEL'] == 1:
        print 'using pretrained vid caption model:%s'%(CFG['MODEL PATH'])
        paramfile = open(CFG['MODEL PATH'],'r')
        net_params = cPickle.load(paramfile)
        paramfile.close()
        model = model_factory.build_vidcaption_model()
        lasagne.layers.set_all_param_values(model[model['net name']]['word_prob'],net_params)
    else:
        print 'build new vidcaption model'
        model = model_factory.build_vidcaption_model()
    model_factory.show_network_configuration(model[model['net name']])
    print 'network computation graph completed!!!'
    print 40 * '*'
    print 'start compiling neccessary experimental functions'
    print 'compiling...'
    exp_func = model_factory.train_test_func(model)

    return model,exp_func

def do_train_exp():
    print 'Training mp_lstm Network'
    print 'Experimental settins:'
    print 'NUMBER EPOCH: %d'%(CFG['NUM_EPOCH'])
    print 'BATCH SIZE: %d'%(CFG['BATCH_SIZE'])
    print 40*'-'

    print 40*'-'
    print 'build model...'
    model,exp_func = exp_model()
    print 'model ok'
    print 40*'*'
    num_samples = len(CFG['TRAIN'])
    num_batch = 0#num_samples / CFG['BATCH_SIZE']
    best_loss = np.inf
    acc=[]
    best_m = 0.
    best_b = 0.
#    print 'training...'
    for iepoch in np.arange(CFG['NUM_EPOCH']):

        epoch_loss = 0.
        epoch_acc =0.
	epoch_att_loss = 0.
	epoch_sem_loss = 0.
        for ibatch in np.arange(num_batch):
            batch_idx = CFG['TRAIN'][ibatch*CFG['BATCH_SIZE']:(ibatch+1)*CFG['BATCH_SIZE']]
            batch_data,batch_words,gt_words,mask,k_words = data.get_batch_data(batch_idx,feat_type='ResNet')
            predict_words = np.reshape(gt_words,(-1,))
         #    print batch_data.shape
         #    print batch_words.shape
         #    print mask.shape
         #    print predict_words.shape
	        # print k_words.shape
         #    word2sent([batch_words[0]])
	        # label2word([k_words[0]])
            print 'forward and backward...'
#            batch_loss,batch_acc,att,att_loss,sem_loss= exp_func['train func'](batch_data,batch_words,mask,predict_words,k_words)
	    try:
            	batch_loss,batch_acc,att,att_loss,sem_loss= exp_func['train func'](batch_data,batch_words,mask,predict_words,k_words)
	    except Exception,e_data:
		        print 'Found Exception'
		        print e_data
		        continue
	    print '%d epoch %d batch: loss %f att loss: %f sem loss: %f acc %f'%(iepoch+1, ibatch+1, batch_loss,att_loss,sem_loss, batch_acc)
            print 'attention:'
#            print att.shape
            print 40*'*'
            print 'attention values from all frames at first step'
            print att[0,0]
            print 40*'*'
            print 'attention values from all time steps at first frame'
            print att[0,:,0]
            epoch_loss += batch_loss
            epoch_acc += batch_acc
	    epoch_att_loss += att_loss
	    epoch_sem_loss += sem_loss
	num_batch+=1
        epoch_loss /= num_batch
        epoch_acc /= num_batch
	epoch_att_loss /= num_batch
	epoch_sem_loss /= num_batch
        train_acc = epoch_acc
        logfile = open('logs/VIDCAP_ATT/log_train_'+time.strftime('%Y-%m-%d',time.localtime(time.time())),'a+')
        print >> logfile,time.strftime('%Y-%m-%d %H:%M:%S',
                                       time.localtime(time.time()))+'\n epoch %d script train loss:%f att loss %f sem loss %f train acc:%f'%(iepoch+1,epoch_loss,epoch_att_loss,epoch_sem_loss,train_acc)
        logfile.close()
        print 'mean batch loss: %f'%epoch_loss
	print 'mean batch att loss: %f'%epoch_att_loss
	print 'mean batch sem loss: %f'%epoch_sem_loss
        print 'mean batch acc: %f'%epoch_acc
        print 40*'-'
        if epoch_loss < best_loss:
            print 'find better training result.'
            print 'saving model'
            net_params = lasagne.layers.get_all_param_values(model[model['net name']]['word_prob'])
            modelfile = open('../models/sta_fg_params.pkl','wb')
            cPickle.dump(net_params,modelfile)
            modelfile.close()
            if iepoch>=0:
                print 'lets validating our model'
                eval_res = eval_model.evaluate(exp_func['sent prob'],exp_func['key word'],'ResNet','valid')
                logfile = open('logs/VIDCAP_ATT/log_train_'+time.strftime('%Y-%m-%d',time.localtime(time.time())),'a+')
		print >> logfile,time.strftime('%Y-%m-%d %H:%M:%S',
                                       time.localtime(time.time()))+ ' evaluation results\n' 
  		if eval_res['METEOR'] > best_m and eval_res['Bleu_4']>best_b:
                    best_m = eval_res['METEOR']
                    best_b = eval_res['Bleu_4']
                    modelfile = open('../models/best_valided_stafg_params.pkl','wb')
                    cPickle.dump(net_params,modelfile)
                    modelfile.close()
                    print >> logfile,'find better model!!!'
		print >> logfile,'validation results:'
                for metric, score in eval_res.items():
                    print >> logfile,'%s: %.3f'%(metric, score),
		print >> logfile,'\n'
		logfile.close()

    # testing model
    print 'training done! Testing model...'
    eval_res = eval_model.evaluate(exp_func['sent prob'],exp_func['key word'],'ResNet','test')
    logfile = open('logs/VIDCAP_ATT/log_test_'+time.strftime('%Y-%m-%d',time.localtime(time.time())),'a+')
    print >> logfile,time.strftime('%Y-%m-%d %H:%M:%S',
                        time.localtime(time.time()))+ ' testing results\n'
    for metric, score in eval_res.items():
        print >> logfile,'%s: %.3f'%(metric, score),
    print >> logfile,'\n'
    logfile.close()
    print 'DONE'
    
def word2sent(wordids):
    print 'Captions:'
    for item in wordids:
        cap = ''
    	c = 0
        for i in item:
	    if c==0:
		c+=1
		continue    
	    if i==0:
		break
            cap = cap+CFG['idx2word'][i]+' '
        print cap
    return

def label2word(labels):
    print 'Key words:'
    idx = np.where(labels==1)
    for item in labels:
	idx = np.where(item==1)
        cap = ''
        for i in idx[0]:
            cap = cap+CFG['idx2word'][i]+' '
        print cap
    return

if __name__ == '__main__':
    do_train_exp()
