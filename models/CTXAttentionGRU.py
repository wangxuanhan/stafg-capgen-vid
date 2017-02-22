# -*- coding: utf-8 -*-
__author__ = 'Xuanhan Wang'
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan

from lasagne.layers import MergeLayer,Layer
# from lasagne.layers import InputLayer
# from lasagne.layers import DenseLayer
# from lasagne.layers import helper
import theano
from theano import tensor as T
import numpy as np

__all__ = [
    "Gate",
    "CTXAttentionGRULayer"
]

class Gate(object):
    def __init__(self, W_in=init.Normal(0.1), W_hid=init.Normal(0.1),
                 W_cell=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        self.W_in = W_in
        self.W_hid = W_hid
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

class CTXAttentionGRULayer(MergeLayer):
    
    def __init__(self, incoming, num_units,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh),
                 attentiongate=Gate(W_cell=None,nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 att_init=init.Constant(0.),
                 hyb_init=init.Constant(0.),
                 ctx_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=False,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        incomings = incoming
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.att_init_incoming_index = -1
        self.hyb_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        if isinstance(att_init, Layer):
            incomings.append(att_init)
            self.att_init_incoming_index = len(incomings)-1
        if isinstance(hyb_init, Layer):
            incomings.append(hyb_init)
            self.hyb_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(CTXAttentionGRULayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        # wang recitifed
        self.attenval = att_init
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]
        # wang
        visual_shape = self.input_shapes[1]
        ctx_shape = self.input_shapes[2]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])
        num_vis_inputs = np.prod(visual_shape[2:])
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        def add_attention_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (visual_shape[2], num_units),
                                   name="v_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_in, (num_units, num_units),
                                   name="h_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_in, (num_units,1),
                                   name="att_in_{}".format(gate_name)),
                    self.add_param(gate.W_in, (ctx_shape[1], num_units),
                                   name="ctx_in_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        # wang
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')

        (self.W_att_v, self.W_att_hid, self.W_att, self.W_ctx,
            self.b_att,self.nonlinearity_attention) = add_attention_gate_params(
             attentiongate, 'attention_gate')

        hybrid_gate = Gate(nonlinearity=nonlinearities.tanh)
        self.W_hybrid = self.add_param(hybrid_gate.W_in,(num_units+num_vis_inputs+num_inputs,num_units),
                                       name='W_hybrid_to_final')
        self.b_hybrid = self.add_param(hybrid_gate.b,(num_units,),
                                       name='b_hybrid_to_final',regularizable=False)
        self.nonlinearity_hybird = hybrid_gate.nonlinearity
        # Initialize hidden state
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

        # wang Initialize attention state
        if isinstance(att_init, Layer):
            self.att_init = att_init
        else:
            self.att_init = self.add_param(
                att_init, (1, visual_shape[1]), name="att_init",
                trainable=learn_init, regularizable=False)
        if isinstance(hyb_init, Layer):
            self.hyb_init = hyb_init
        else:
            self.hyb_init = self.add_param(
                hyb_init, (1, num_units), name="hyb_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units*1
    # wang rectified
    def get_output_attention(self):
        return self.attenval
    def get_output_hidden(self):
        return self.hid_out
    def get_output_for(self, inputs, **kwargs):
        
        # Retrieve the layer input
        input = inputs[0]
        # wang Retrieve the Visual input
        vis_input = inputs[1]
        global_ctx = inputs[2]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None

        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        att_init = None
        hyb_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.att_init_incoming_index > 0:
            att_init = inputs[self.att_init_incoming_index]
        if self.hyb_init_incoming_index > 0:
            hyb_init = inputs[self.hyb_init_incoming_index]
        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

        # When theano.scan calls step, input_n will be (n_batch, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input

        def vis_attention(pre_hid,v,global_ctx):
            # compute attention values
            # cur_x = (n_batch, n_word_features)
            # v = (n_batch, n_visual_sequence, n_visual_feature)
            # pre_hid = (n_batch, num_units)
            # global_ctx = (n_batch, num_units)
            # W_att_hid (num_units, num_units)
            # W_att_v = (n_visual_feature, num_units)
            # W_att =(num_units,1)
            pre_hid_state = T.dot(pre_hid, self.W_att_hid)
            vis_state = T.dot(v, self.W_att_v)
            ctx_state = T.dot(global_ctx,self.W_ctx)
            a_state = pre_hid_state[:,None,:] + vis_state +\
                      ctx_state[:,None,:]+self.b_att
            a_state = self.nonlinearity_attention(a_state)
            a_state = T.dot(a_state,self.W_att)
            a_state = T.reshape(a_state,(-1,a_state.shape[1]))
            att = T.nnet.softmax(a_state)
            return att

        def step(input_n, hid_previous, att_previous,hyb_previous, *args):
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)
            word_emb = input_n
            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)
            # TEST
            att = vis_attention(hid_previous,vis_input,global_ctx)
            ctx_vis_feats = T.sum(vis_input * att[:,:,None],axis=1).reshape((-1,vis_input.shape[2])) # (n_batch, n_vis_feat)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Reset and update gates
            # wang
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            # wang
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate*hidden_update_hid
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate)*hid_previous + updategate*hidden_update

            _feats = T.concatenate([hid,ctx_vis_feats,global_ctx],axis=1)
            hybrid_feats = T.dot(_feats, self.W_hybrid) + self.b_hybrid
            hybrid_feats = self.nonlinearity_hybird(hybrid_feats)
            return [hid,att,hybrid_feats]

        def step_masked(input_n, mask_n, hid_previous, att_previous, hyb_previous,*args):
            hid,att,hyb_feats = step(input_n, hid_previous,att_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = T.switch(mask_n, hid, hid_previous)
            # not_mask = 1 - mask_n
            # att = att*mask_n + att_previous*not_mask
            # hid = hid*mask_n + hid_previous*not_mask
            att = T.switch(mask_n, att, att_previous)
            hyb_feats = T.switch(mask_n,hyb_feats,hyb_previous)
            return [hid, att, hyb_feats]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)
        # wang
        if not isinstance(self.att_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_att)
            att_init = T.dot(T.ones((num_batch, 1)), self.att_init)

        if not isinstance(self.hyb_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hyb_init = T.dot(T.ones((num_batch, 1)), self.hyb_init)
        # The hidden-to-hidden weight matrix is always used in step

        non_seqs = [W_hid_stacked,vis_input,global_ctx]+[self.W_att,self.W_att_hid,self.W_att_v,self.b_att]+ \
                    [self.W_hybrid,self.b_hybrid] +[self.W_ctx]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out, att_out,hyb_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init,att_init,hyb_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out, att_out, hyb_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init,att_init,hyb_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
            att_out = att_out[-1]
            hyb_out = hyb_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            att_out = att_out.dimshuffle(1, 0, 2)
            hyb_out = hyb_out.dimshuffle(1, 0, 2)
            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]
                att_out = att_out[:, ::-1]
                hyb_out = hyb_out[:, ::-1]
        self.attenval = att_out
        self.hid_out = hid_out
        return hyb_out
