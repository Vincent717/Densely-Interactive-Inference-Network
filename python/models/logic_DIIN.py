import tensorflow as tf
from util import blocks
from my.tensorflow.nn import softsel, get_logits, highway_network, multi_conv1d, linear, conv2d, cosine_similarity, variable_summaries, dense_logits, fuse_gate
from my.tensorflow import flatten, reconstruct, add_wd, exp_mask
import numpy as np

class MyModel(object):
    def __init__(self, config, seq_length, emb_dim, hidden_dim, emb_train, embeddings = None, pred_size = 3, context_seq_len = None, query_seq_len = None):
        ## Define hyperparameters
        # tf.reset_default_graph()
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.sequence_length = seq_length
        self.pred_size = pred_size 
        self.context_seq_len = context_seq_len
        self.query_seq_len = query_seq_len
        # self.config = config

        ## Define the placeholders    
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='premise')
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='hypothesis')
        self.premise_pos = tf.placeholder(tf.int32, [None, self.sequence_length, 47], name='premise_pos')
        self.hypothesis_pos = tf.placeholder(tf.int32, [None, self.sequence_length, 47], name='hypothesis_pos')
        self.premise_char = tf.placeholder(tf.int32, [None, self.sequence_length, config.char_in_word_size], name='premise_char')
        self.hypothesis_char = tf.placeholder(tf.int32, [None, self.sequence_length, config.char_in_word_size], name='hypothesis_char')
        self.premise_exact_match = tf.placeholder(tf.int32, [None, self.sequence_length,1], name='premise_exact_match')
        self.hypothesis_exact_match = tf.placeholder(tf.int32, [None, self.sequence_length,1], name='hypothesis_exact_match')
        self.premise_dependency = tf.placeholder(tf.int32, [None, self.sequence_length, config.depend_size], name='premise_dependency')
        self.hypothesis_dependency = tf.placeholder(tf.int32, [None, self.sequence_length, config.depend_size], name='hypothesis_dependency')

        self.and_index = tf.placeholder(tf.int32, [None,], name='and_index')
        #self.epoch = tf.placeholder(tf.int32, [1], name='epoch')
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        
        self.dropout_keep_rate = tf.train.exponential_decay(config.keep_rate, self.global_step, config.dropout_decay_step, config.dropout_decay_rate, staircase=False, name='dropout_keep_rate')
        config.keep_rate = self.dropout_keep_rate
        tf.summary.scalar('dropout_keep_rate', self.dropout_keep_rate)

        self.y = tf.placeholder(tf.int32, [None], name='label_y')
        self.keep_rate_ph = tf.placeholder(tf.float32, [], name='keep_prob')
        self.is_train = tf.placeholder('bool', [], name='is_train')
        
        ## Fucntion for embedding lookup and dropout at embedding layer
        def emb_drop(E, x):
            emb = tf.nn.embedding_lookup(E, x)
            emb_drop = tf.cond(self.is_train, lambda: tf.nn.dropout(emb, config.keep_rate), lambda: emb)
            return emb_drop

        # Get lengths of unpadded sentences    
        prem_seq_lengths, prem_mask = blocks.length(self.premise_x)  # mask [N, L , 1]
        hyp_seq_lengths, hyp_mask = blocks.length(self.hypothesis_x)
        self.prem_mask = prem_mask
        self.hyp_mask = hyp_mask


        ### Embedding layer ###
        with tf.variable_scope("emb"):
            with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                self.E = tf.Variable(embeddings, trainable=emb_train)
                premise_in = emb_drop(self.E, self.premise_x)   #P
                hypothesis_in = emb_drop(self.E, self.hypothesis_x)  #H
    
        with tf.variable_scope("char_emb"):
            char_emb_mat = tf.get_variable("char_emb_mat", shape=[config.char_vocab_size, config.char_emb_size])
            with tf.variable_scope("char") as scope:
                char_pre = tf.nn.embedding_lookup(char_emb_mat, self.premise_char)
                char_hyp = tf.nn.embedding_lookup(char_emb_mat, self.hypothesis_char)

                filter_sizes = list(map(int, config.out_channel_dims.split(','))) #[100]
                heights = list(map(int, config.filter_heights.split(',')))        #[5]
                assert sum(filter_sizes) == config.char_out_size, (filter_sizes, config.char_out_size)
                with tf.variable_scope("conv") as scope:
                    conv_pre = multi_conv1d(char_pre, filter_sizes, heights, "VALID", self.is_train, config.keep_rate, scope='conv')
                    scope.reuse_variables()  
                    conv_hyp = multi_conv1d(char_hyp, filter_sizes, heights, "VALID", self.is_train, config.keep_rate, scope='conv')
                    conv_pre = tf.reshape(conv_pre, [-1, self.sequence_length, config.char_out_size])
                    conv_hyp = tf.reshape(conv_hyp, [-1, self.sequence_length, config.char_out_size])
            premise_in = tf.concat([premise_in, conv_pre], axis=2)
            hypothesis_in = tf.concat([hypothesis_in, conv_hyp], axis=2)


        # syntatic imformation
        premise_in = tf.concat((premise_in, tf.cast(self.premise_pos, tf.float32)), axis=2)
        hypothesis_in = tf.concat((hypothesis_in, tf.cast(self.hypothesis_pos, tf.float32)), axis=2)

        premise_in = tf.concat([premise_in, tf.cast(self.premise_exact_match, tf.float32)], axis=2)
        hypothesis_in = tf.concat([hypothesis_in, tf.cast(self.hypothesis_exact_match, tf.float32)], axis=2)


        with tf.variable_scope("highway") as scope:
            premise_in = highway_network(premise_in, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)    
            scope.reuse_variables()
            hypothesis_in = highway_network(hypothesis_in, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)


        ## self attention process
        def model_self_attention(config, premise_in, hypothesis_in, prem_mask, hyp_mask):
            pre = premise_in
            hyp = hypothesis_in
            for i in range(config.self_att_enc_layers):
                with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                    p = self_attention_layer(config, self.is_train, pre, p_mask=prem_mask, scope="{}_layer_self_att_enc".format(i)) # [N, len, dim]    
                    h = self_attention_layer(config, self.is_train, hyp, p_mask=hyp_mask, scope="{}_layer_self_att_enc_h".format(i))
                    pre = p
                    hyp = h
                    variable_summaries(p, "p_self_enc_summary_layer_{}".format(i))
                    variable_summaries(h, "h_self_enc_summary_layer_{}".format(i))
          
            if config.use_depend:
                pre1 = p
                hyp1 = h
                for i in range(config.denp_enc_layers):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                        if config.use_depend:
                            p1 = dependency_layer(config, self.is_train, pre1, self.premise_dependency, p_mask=prem_mask, scope="{}_layer_dependency_enc".format(i))
                            h1 = dependency_layer(config, self.is_train, hyp1, self.hypothesis_dependency, p_mask=hyp_mask, scope="{}_layer_dependency_enc_h".format(i))
                        pre1 = p1
                        hyp1 = h1
                        variable_summaries(p, "p_denp_enc_summary_layer_{}".format(i))
                        variable_summaries(h, "h_denp_enc_summary_layer_{}".format(i))
            
                p = tf.concat([p, p1], -1) 
                h = tf.concat([h, h1], -1)
            return p, h 

        ## main process : interaction + dense net
        def model_one_side(config, main, support, main_length, support_length, main_mask, support_mask, scope):
            bi_att_mx = bi_attention_mx(config, self.is_train, main, support, p_mask=main_mask, h_mask=support_mask, sequence_length=self.sequence_length) # [N, PL, HL]
           
            bi_att_mx = tf.cond(self.is_train, lambda: tf.nn.dropout(bi_att_mx, config.keep_rate), lambda: bi_att_mx)
            out_final = dense_net(config, bi_att_mx, self.is_train)
            
            return out_final

        # self attention
        with tf.variable_scope("prepro") as scope:
            p, h = model_self_attention(config, premise_in, hypothesis_in, prem_mask, hyp_mask)

        # main
        with tf.variable_scope("main") as scope:
            premise_final = model_one_side(config, p, h, prem_seq_lengths, hyp_seq_lengths, prem_mask, hyp_mask, scope="premise_as_main")
            f0 = premise_final

            self.logits = linear(f0, self.pred_size ,True, bias_start=0.0, scope="logit", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate,
                                is_train=self.is_train)

        tf.summary.histogram('logit_histogram', self.logits)

        ## Hu 2016
        if config.use_logic:
            minus_one = tf.Variable(-1)

            def go_through_whole_model(premise_in, hypothesis_in, config=config, prem_mask=prem_mask, hyp_mask=hyp_mask, pred_size=self.pred_size, is_train=self.is_train):
                # with tf.variable_scope("highway") as scope:
                #     premise_in = highway_network(premise_in, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)    
                #     scope.reuse_variables()
                #     hypothesis_in = highway_network(hypothesis_in, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)

                # self attention
                #with tf.variable_scope("prepro") as scope:
                #    p, h = model_self_attention(config, premise_in, hypothesis_in, prem_mask, hyp_mask)

                p, h = premise_in, hypothesis_in
                # main
                with tf.variable_scope("main", reuse=True) as scope:
                    premise_final = model_one_side(config, p, h, prem_seq_lengths, hyp_seq_lengths, prem_mask, hyp_mask, scope="premise_as_main")
                    f0 = premise_final

                    logits = linear(f0, pred_size ,True, bias_start=0.0, scope="logit", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate,
                                        is_train=is_train)
                    logits = tf.nn.softmax(logits)
                return logits

            def cal_and_distr(sub_logits1, sub_logits2, c, and_mask, lambdal=1.):
                """
                there are two rules:
                AE: 1(y=Entailment) -> (p1_E V p2_E) ^ (p1_E V p2_E) -> 1(y=E)
                AC: 1(y=Contradiction) -> (p1_C V p2_C) ^ (p1_C V p2_C) -> 1(y=C)
                """
                # for rule in rules:
                #     if rule == 'AndE':
                p1_add_p2 = sub_logits1 + sub_logits2
                pre_distr = tf.minimum(p1_add_p2, 1)  # 70x3
                r_AE_y0 = (pre_distr[:,0] + 1) / 2  # 70x1
                r_AC_y0 = (2 - pre_distr[:,2]) / 2  # 70x1
                r_AN_y0 = 1
                r_AE_y1 = (2 - pre_distr[:,0]) / 2
                r_AC_y1 = (pre_distr[:,2] + 1) / 2
                r_AN_y1 =  p1_add_p2[:,1] / 2
                r_AE_y2 = r_AE_y1
                r_AC_y2 = r_AC_y0
                r_AN_y2 = 1

                r_y0 = c*lambdal* ( 3. - r_AE_y0 - r_AC_y0 - r_AN_y0)  # 70x1
                r_y1 = c*lambdal* ( 3. - r_AE_y1 - r_AC_y1 - r_AN_y1)  # 70x1
                r_y2 = c*lambdal* ( 3. - r_AE_y2 - r_AC_y2 - r_AN_y2)  # 70x1
                r_y0 = tf.where(tf.equal(and_mask, -1), tf.zeros_like(r_y0), r_y0)  # mask
                r_y1 = tf.where(tf.equal(and_mask, -1), tf.zeros_like(r_y1), r_y1)
                r_y2 = tf.where(tf.equal(and_mask, -1), tf.zeros_like(r_y2), r_y2)
                r_y0 = tf.reshape(r_y0, [-1, 1])
                r_y1 = tf.reshape(r_y1, [-1, 1])
                r_y2 = tf.reshape(r_y2, [-1, 1])
                result = - tf.concat([r_y0, r_y1, r_y2], axis=1)
                # tuncate
                #distr_y0 = distr_all[:,0]
                #distr_y0 = distr_y0.reshape([distr_y0.shape[0], 1])
                #distr_y0_copies = tf.tile(distr_y0, [1, result.shape[1]])
                #result -= distr_y0_copies
                result = tf.maximum(tf.minimum(result, 60.), -60.)

                return result


            def slice_full(index, p):
                p1_ = tf.slice(p, [0,0], [index, -1])
                p2_ = tf.slice(p, [index,0], [-1, -1])
                p1_full = tf.concat([p1_, tf.zeros_like(p2_)], axis=0)
                p2_full = tf.concat([p2_, tf.zeros_like(p1_)], axis=0)
                #p1_full = tf.reshape(p1_full, [-1, self.sequence_length, p1_full.shape[-1]])
                #p2_full = tf.reshape(p2_full, [-1, self.sequence_length, p2_full.shape[-1]])
                return p1_full, p2_full

            def two_zero(p):
                return tf.zeros_like(p), tf.zeros_like(p)

            def slice_it_on(elems):
                """
                index : ? x 1
                p     : ? x 48 x 448
                h     : ? x 48 x 448
                since it is map_fn, so ? will be ignored
                """
                index, p, h = elems
                #index = index[0]
                p1, p2 = tf.cond(tf.equal(index, minus_one), lambda: two_zero(p), lambda: slice_full(index, p))
                sub_h = tf.cond(tf.equal(index, minus_one), lambda: tf.zeros_like(h), lambda: h)
                return p1, p2, sub_h

            # construct teacher network output
            q_y_x = self.logits

            p1, p2, sub_h = tf.map_fn(slice_it_on, (self.and_index, p, h), dtype=(tf.float32, tf.float32, tf.float32))
            sub_logits1 = go_through_whole_model(p1, sub_h)
            sub_logits2 = go_through_whole_model(p2, sub_h)
            c = tf.constant(config.C , dtype=tf.float32, shape=[], name='c')
            lambdal = tf.constant(config.lambdal , dtype=tf.float32, shape=[], name='lambdal')
            distr = tf.exp(cal_and_distr(sub_logits1, sub_logits2, c, self.and_index, lambdal))
            q_y_x = q_y_x * distr
            self.q_y_x = q_y_x


        # Define the cost function
        if not config.use_logic:
            self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, dimension=1),tf.cast(self.y,tf.int64)), tf.float32))
            tf.summary.scalar('acc', self.acc)
            tf.summary.scalar('loss', self.total_cost)
            #self.auc_ROC = tf.metrics.auc(tf.cast(self.y,tf.int64), tf.arg_max(self.logits, dimension=1), curve = 'ROC')
            #self.auc_PR =  tf.metrics.auc(tf.cast(self.y,tf.int64), tf.arg_max(self.logits, dimension=1), curve = 'PR')
            #tf.summary.scalar('auc_ROC', self.auc_ROC)
            #tf.summary.scalar('auc_PR', self.auc_PR)
            # calculate acc 
        else:
            get_pi = lambda x, y: x * 0.9**tf.cast(y/6750, tf.float32)
            pi = get_pi(config.pi, self.global_step)
            self.total_cost = (1-pi)*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
            self.total_cost += pi*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.arg_max(q_y_x, dimension=1), logits=self.logits))
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, dimension=1),tf.cast(self.y,tf.int64)), tf.float32))
            tf.summary.scalar('acc', self.acc)
            tf.summary.scalar('loss', self.total_cost)
        
        # L2 Loss
        if config.l2_loss:
            if config.sigmoid_growing_l2loss:
                weights_added = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() if tensor.name.endswith("weights:0") and not tensor.name.endswith("weighted_sum/weights:0") or tensor.name.endswith('kernel:0')])
                full_l2_step = tf.constant(config.weight_l2loss_step_full_reg , dtype=tf.int32, shape=[], name='full_l2reg_step')
                full_l2_ratio = tf.constant(config.l2_regularization_ratio , dtype=tf.float32, shape=[], name='l2_regularization_ratio')
                gs_flt = tf.cast(self.global_step , tf.float32)
                half_l2_step_flt = tf.cast(full_l2_step / 2 ,tf.float32)

                # (self.global_step - full_l2_step / 2)
                # tf.cast((self.global_step - full_l2_step / 2) * 8, tf.float32) / tf.cast(full_l2_step / 2 ,tf.float32)
                # l2loss_ratio = tf.sigmoid( tf.cast((self.global_step - full_l2_step / 2) * 8, tf.float32) / tf.cast(full_l2_step / 2 ,tf.float32)) * full_l2_ratio
                l2loss_ratio = tf.sigmoid( ((gs_flt - half_l2_step_flt) * 8) / half_l2_step_flt) * full_l2_ratio
                tf.summary.scalar('l2loss_ratio', l2loss_ratio)
                l2loss = weights_added * l2loss_ratio
            else:
                l2loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() if tensor.name.endswith("weights:0") or tensor.name.endswith('kernel:0')]) * tf.constant(config.l2_regularization_ratio , dtype='float', shape=[], name='l2_regularization_ratio')
            tf.summary.scalar('l2loss', l2loss)
            self.total_cost += l2loss

        # semantic Loss
        if config.semantic_loss:
            #semantic_loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() 
            #    if tensor.name.endswith("weights:0") or tensor.name.endswith('kernel:0')]) 
            
            def cal_exactly_one_loss(logits):
                #semantic_loss = tf.Variable(tf.zeros([], dtype=np.float32), name='semantic_loss_term')

                return tf.reduce_sum(-tf.log(logits[:,0]*(1-logits[:,1])*(1-logits[:,2]) +
                               logits[:,1]*(1-logits[:,0])*(1-logits[:,2]) +
                               logits[:,2]*(1-logits[:,0])*(1-logits[:,1])
                                ))

            def cal_logic_rules_loss(rules, logits):
                def cal_logic_rule(ro, ls):
                    if ro == '0':
                        return ls[0]*(1-ls[1])*(1-ls[2])
                    elif ro == '1':
                        return ls[1]*(1-ls[0])*(1-ls[2])
                    elif ro == '2':
                        return ls[2]*(1-ls[0])*(1-ls[1])
                return -tf.log(tf.add_n([cal_logic_rule(rule_output, logits) for rule_output in rules]))

            #semantic_loss = cal_logic_rules_loss(self.rules_output, self.logits)
            if config.use_exactly_one:
                semantic_loss = cal_exactly_one_loss(self.logits)
            semantic_loss = tf.reduce_mean(semantic_loss)
            semantic_loss = semantic_loss * tf.constant(config.semantic_regularization_ratio , dtype='float', shape=[], name='semantic_regularization_ratio')
            tf.summary.scalar('semantic loss', semantic_loss)
            self.total_cost += semantic_loss


        if config.wo_enc_sharing or config.wo_highway_sharing_but_penalize_diff:
            diffs = []
            for i in range(config.self_att_enc_layers):
                for tensor in tf.trainable_variables():
                    print(tensor.name)
                    if tensor.name == "prepro/{}_layer_self_att_enc/self_attention/h_logits/first/kernel:0".format(i):
                        l_lg = tensor 
                    elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_attention/h_logits/first/kernel:0".format(i):
                        r_lg = tensor 
                    elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/lhs_1/kernel:0".format(i):    
                        l_fg_lhs_1 = tensor 
                    elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/lhs_1/kernel:0".format(i):
                        r_fg_lhs_1= tensor
                    elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/rhs_1/kernel:0".format(i):
                        l_fg_rhs_1= tensor
                    elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/rhs_1/kernel:0".format(i):
                        r_fg_rhs_1= tensor
                    elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/lhs_2/kernel:0".format(i):
                        l_fg_lhs_2= tensor
                    elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/lhs_2/kernel:0".format(i):
                        r_fg_lhs_2= tensor
                    elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/rhs_2/kernel:0".format(i):
                        l_fg_rhs_2= tensor
                    elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/rhs_2/kernel:0".format(i):
                        r_fg_rhs_2= tensor

                    if config.two_gate_fuse_gate:
                        if tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/lhs_3/kernel:0".format(i):    
                            l_fg_lhs_3 = tensor 
                        elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/lhs_3/kernel:0".format(i):
                            r_fg_lhs_3 = tensor
                        elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/rhs_3/kernel:0".format(i):
                            l_fg_rhs_3 = tensor
                        elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/rhs_3/kernel:0".format(i):
                            r_fg_rhs_3 = tensor

                diffs += [l_lg - r_lg, l_fg_lhs_1 - r_fg_lhs_1, l_fg_rhs_1 - r_fg_rhs_1, l_fg_lhs_2 - r_fg_lhs_2, l_fg_rhs_2 - r_fg_rhs_2]
                if config.two_gate_fuse_gate:
                    diffs += [l_fg_lhs_3 - r_fg_lhs_3, l_fg_rhs_3 - r_fg_rhs_3]
            

            diff_loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in diffs]) * tf.constant(config.diff_penalty_loss_ratio , dtype='float', shape=[], name='diff_penalty_loss_ratio')
            tf.summary.scalar('diff_penalty_loss', diff_loss)
            self.total_cost += diff_loss


        self.summary = tf.summary.merge_all()

        total_parameters = 0
        for v in tf.global_variables():
            if not v.name.endswith("weights:0") and not v.name.endswith("biases:0") and not v.name.endswith('kernel:0') and not v.name.endswith('bias:0'):
                continue
            print(v.name)
            # print(type(v.name))
            shape = v.get_shape().as_list()
            param_num = 1
            for dim in shape:
                param_num *= dim 
            print(param_num)
            total_parameters += param_num
        print(total_parameters)

class MyModelWn(object):
    def __init__(self, config, seq_length, emb_dim, hidden_dim, emb_train, embeddings = None, pred_size = 3, context_seq_len = None, query_seq_len = None):
        ## Define hyperparameters
        # tf.reset_default_graph()
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.sequence_length = seq_length
        self.pred_size = pred_size 
        self.context_seq_len = context_seq_len
        self.query_seq_len = query_seq_len
        # self.config = config

        ## Define the placeholders    
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='premise')
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='hypothesis')
        self.premise_pos = tf.placeholder(tf.int32, [None, self.sequence_length, 47], name='premise_pos')
        self.hypothesis_pos = tf.placeholder(tf.int32, [None, self.sequence_length, 47], name='hypothesis_pos')
        self.premise_char = tf.placeholder(tf.int32, [None, self.sequence_length, config.char_in_word_size], name='premise_char')
        self.hypothesis_char = tf.placeholder(tf.int32, [None, self.sequence_length, config.char_in_word_size], name='hypothesis_char')
        self.premise_exact_match = tf.placeholder(tf.int32, [None, self.sequence_length,1], name='premise_exact_match')
        self.hypothesis_exact_match = tf.placeholder(tf.int32, [None, self.sequence_length,1], name='hypothesis_exact_match')
        self.wordnet_rel = tf.placeholder(tf.float32, [None, self.sequence_length, self.sequence_length, 5], name='wordnet_rel')
        self.premise_dependency = tf.placeholder(tf.int32, [None, self.sequence_length, config.depend_size], name='premise_dependency')
        self.hypothesis_dependency = tf.placeholder(tf.int32, [None, self.sequence_length, config.depend_size], name='hypothesis_dependency')

        self.and_index = tf.placeholder(tf.int32, [None,], name='and_index')
        #self.epoch = tf.placeholder(tf,int32, [1], name='epoch')


        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        
        self.dropout_keep_rate = tf.train.exponential_decay(config.keep_rate, self.global_step, config.dropout_decay_step, config.dropout_decay_rate, staircase=False, name='dropout_keep_rate')
        config.keep_rate = self.dropout_keep_rate
        tf.summary.scalar('dropout_keep_rate', self.dropout_keep_rate)

        self.y = tf.placeholder(tf.int32, [None], name='label_y')
        self.keep_rate_ph = tf.placeholder(tf.float32, [], name='keep_prob')
        self.is_train = tf.placeholder('bool', [], name='is_train')
        
        ## Fucntion for embedding lookup and dropout at embedding layer
        def emb_drop(E, x):
            emb = tf.nn.embedding_lookup(E, x)
            emb_drop = tf.cond(self.is_train, lambda: tf.nn.dropout(emb, config.keep_rate), lambda: emb)
            return emb_drop

        # Get lengths of unpadded sentences    
        prem_seq_lengths, prem_mask = blocks.length(self.premise_x)  # mask [N, L , 1]
        hyp_seq_lengths, hyp_mask = blocks.length(self.hypothesis_x)
        self.prem_mask = prem_mask
        self.hyp_mask = hyp_mask


        ### Embedding layer ###
        with tf.variable_scope("emb"):
            with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                self.E = tf.Variable(embeddings, trainable=emb_train)
                premise_in = emb_drop(self.E, self.premise_x)   #P
                hypothesis_in = emb_drop(self.E, self.hypothesis_x)  #H
    
        with tf.variable_scope("char_emb"):
            char_emb_mat = tf.get_variable("char_emb_mat", shape=[config.char_vocab_size, config.char_emb_size])
            with tf.variable_scope("char") as scope:
                char_pre = tf.nn.embedding_lookup(char_emb_mat, self.premise_char)
                char_hyp = tf.nn.embedding_lookup(char_emb_mat, self.hypothesis_char)

                filter_sizes = list(map(int, config.out_channel_dims.split(','))) #[100]
                heights = list(map(int, config.filter_heights.split(',')))        #[5]
                assert sum(filter_sizes) == config.char_out_size, (filter_sizes, config.char_out_size)
                with tf.variable_scope("conv") as scope:
                    conv_pre = multi_conv1d(char_pre, filter_sizes, heights, "VALID", self.is_train, config.keep_rate, scope='conv')
                    scope.reuse_variables()  
                    conv_hyp = multi_conv1d(char_hyp, filter_sizes, heights, "VALID", self.is_train, config.keep_rate, scope='conv')
                    conv_pre = tf.reshape(conv_pre, [-1, self.sequence_length, config.char_out_size])
                    conv_hyp = tf.reshape(conv_hyp, [-1, self.sequence_length, config.char_out_size])
            premise_in = tf.concat([premise_in, conv_pre], axis=2)
            hypothesis_in = tf.concat([hypothesis_in, conv_hyp], axis=2)


        # syntatic imformation
        premise_in = tf.concat((premise_in, tf.cast(self.premise_pos, tf.float32)), axis=2)
        hypothesis_in = tf.concat((hypothesis_in, tf.cast(self.hypothesis_pos, tf.float32)), axis=2)

        premise_in = tf.concat([premise_in, tf.cast(self.premise_exact_match, tf.float32)], axis=2)
        hypothesis_in = tf.concat([hypothesis_in, tf.cast(self.hypothesis_exact_match, tf.float32)], axis=2)


        with tf.variable_scope("highway") as scope:
            premise_in = highway_network(premise_in, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)    
            scope.reuse_variables()
            hypothesis_in = highway_network(hypothesis_in, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)

        ## self attention process
        def model_self_attention(config, premise_in, hypothesis_in, prem_mask, hyp_mask):
            pre = premise_in
            hyp = hypothesis_in
            for i in range(config.self_att_enc_layers):
                with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                    p = self_attention_layer(config, self.is_train, pre, p_mask=prem_mask, scope="{}_layer_self_att_enc".format(i)) # [N, len, dim]    
                    h = self_attention_layer(config, self.is_train, hyp, p_mask=hyp_mask, scope="{}_layer_self_att_enc_h".format(i))
                    pre = p
                    hyp = h
                    variable_summaries(p, "p_self_enc_summary_layer_{}".format(i))
                    variable_summaries(h, "h_self_enc_summary_layer_{}".format(i))
          
            if config.use_depend:
                pre1 = p
                hyp1 = h
                for i in range(config.denp_enc_layers):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                        if config.use_depend:
                            p1 = dependency_layer(config, self.is_train, pre1, self.premise_dependency, p_mask=prem_mask, scope="{}_layer_dependency_enc".format(i))
                            h1 = dependency_layer(config, self.is_train, hyp1, self.hypothesis_dependency, p_mask=hyp_mask, scope="{}_layer_dependency_enc_h".format(i))
                        pre1 = p1
                        hyp1 = h1
                        variable_summaries(p, "p_denp_enc_summary_layer_{}".format(i))
                        variable_summaries(h, "h_denp_enc_summary_layer_{}".format(i))
            
                p = tf.concat([p, p1], -1) 
                h = tf.concat([h, h1], -1)
            return p, h 

        ## main process : interaction + dense net
        def model_one_side(config, main, support, main_length, support_length, main_mask, support_mask, scope):
            bi_att_mx = bi_attention_mx(config, self.is_train, main, support, p_mask=main_mask, h_mask=support_mask, sequence_length=self.sequence_length) # [N, PL, HL]
           
            bi_att_mx = tf.cond(self.is_train, lambda: tf.nn.dropout(bi_att_mx, config.keep_rate), lambda: bi_att_mx)
            out_final = dense_net(config, bi_att_mx, self.is_train)
            
            return out_final


        # self attention
        with tf.variable_scope("prepro") as scope:
            p, h = model_self_attention(config, premise_in, hypothesis_in, prem_mask, hyp_mask)
            
        # main
        with tf.variable_scope("main") as scope:
            premise_final = model_one_side(config, p, h, prem_seq_lengths, hyp_seq_lengths, prem_mask, hyp_mask, scope="premise_as_main")
            f0 = premise_final

            self.logits = linear(f0, self.pred_size ,True, bias_start=0.0, scope="logit", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate,
                                is_train=self.is_train)

        tf.summary.histogram('logit_histogram', self.logits)


        ## Hu 2016
        if config.use_logic:
            minus_one = tf.Variable(-1)

            def go_through_whole_model(premise_in, hypothesis_in, config=config, prem_mask=prem_mask, hyp_mask=hyp_mask, pred_size=self.pred_size, is_train=self.is_train):
                # with tf.variable_scope("highway") as scope:
                #     premise_in = highway_network(premise_in, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)    
                #     scope.reuse_variables()
                #     hypothesis_in = highway_network(hypothesis_in, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)

                # self attention
                #with tf.variable_scope("prepro") as scope:
                #    p, h = model_self_attention(config, premise_in, hypothesis_in, prem_mask, hyp_mask)

                p, h = premise_in, hypothesis_in
                # main
                with tf.variable_scope("main", reuse=True) as scope:
                    premise_final = model_one_side(config, p, h, prem_seq_lengths, hyp_seq_lengths, prem_mask, hyp_mask, scope="premise_as_main")
                    f0 = premise_final

                    logits = linear(f0, pred_size ,True, bias_start=0.0, scope="logit", squeeze=False, wd=config.wd, input_keep_prob=config.keep_rate,
                                        is_train=is_train)
                    logits = tf.nn.softmax(logits)
                return logits

            def cal_and_distr(sub_logits1, sub_logits2, c, and_mask, lambdal=1.):
                """
                there are two rules:
                AE: 1(y=Entailment) -> (p1_E V p2_E) ^ (p1_E V p2_E) -> 1(y=E)
                AC: 1(y=Contradiction) -> (p1_C V p2_C) ^ (p1_C V p2_C) -> 1(y=C)
                """
                # for rule in rules:
                #     if rule == 'AndE':
                p1_add_p2 = sub_logits1 + sub_logits2
                pre_distr = tf.minimum(p1_add_p2, 1)  # 70x3
                r_AE_y0 = (pre_distr[:,0] + 1) / 2  # 70x1
                r_AC_y0 = (2 - pre_distr[:,2]) / 2  # 70x1
                r_AN_y0 = 1
                r_AE_y1 = (2 - pre_distr[:,0]) / 2
                r_AC_y1 = (pre_distr[:,2] + 1) / 2
                r_AN_y1 =  p1_add_p2[:,1] / 2
                r_AE_y2 = r_AE_y1
                r_AC_y2 = r_AC_y0
                r_AN_y2 = 1

                r_y0 = c*lambdal* ( 3. - r_AE_y0 - r_AC_y0 - r_AN_y0)  # 70x1
                r_y1 = c*lambdal* ( 3. - r_AE_y1 - r_AC_y1 - r_AN_y1)  # 70x1
                r_y2 = c*lambdal* ( 3. - r_AE_y2 - r_AC_y2 - r_AN_y2)  # 70x1
                r_y0 = tf.where(tf.equal(and_mask, -1), tf.zeros_like(r_y0), r_y0)  # mask
                r_y1 = tf.where(tf.equal(and_mask, -1), tf.zeros_like(r_y1), r_y1)
                r_y2 = tf.where(tf.equal(and_mask, -1), tf.zeros_like(r_y2), r_y2)
                r_y0 = tf.reshape(r_y0, [-1, 1])
                r_y1 = tf.reshape(r_y1, [-1, 1])
                r_y2 = tf.reshape(r_y2, [-1, 1])
                result = - tf.concat([r_y0, r_y1, r_y2], axis=1)
                # tuncate
                #distr_y0 = distr_all[:,0]
                #distr_y0 = distr_y0.reshape([distr_y0.shape[0], 1])
                #distr_y0_copies = tf.tile(distr_y0, [1, result.shape[1]])
                #result -= distr_y0_copies
                result = tf.maximum(tf.minimum(result, 60.), -60.)

                return result


            def slice_full(index, p):
                p1_ = tf.slice(p, [0,0], [index, -1])
                p2_ = tf.slice(p, [index,0], [-1, -1])
                p1_full = tf.concat([p1_, tf.zeros_like(p2_)], axis=0)
                p2_full = tf.concat([p2_, tf.zeros_like(p1_)], axis=0)
                return p1_full, p2_full

            def two_zero(p):
                return tf.zeros_like(p), tf.zeros_like(p)

            def slice_it_on(elems):
                """
                index : ? x 1
                p     : ? x 48 x 448
                h     : ? x 48 x 448
                since it is map_fn, so ? will be ignored
                """
                index, p, h = elems
                #index = index[0]
                p1, p2 = tf.cond(tf.equal(index, minus_one), lambda: two_zero(p), lambda: slice_full(index, p))
                sub_h = tf.cond(tf.equal(index, minus_one), lambda: tf.zeros_like(h), lambda: h)
                return p1, p2, sub_h

            # construct teacher network output
            q_y_x = self.logits

            p1, p2, sub_h = tf.map_fn(slice_it_on, (self.and_index, p, h), dtype=(tf.float32, tf.float32, tf.float32))
            sub_logits1 = go_through_whole_model(p1, sub_h)
            sub_logits2 = go_through_whole_model(p2, sub_h)
            c = tf.constant(config.C , dtype=tf.float32, shape=[], name='c')
            lambdal = tf.constant(config.lambdal , dtype=tf.float32, shape=[], name='lambdal')
            distr = tf.exp(cal_and_distr(sub_logits1, sub_logits2, c, self.and_index, lambdal))
            q_y_x = q_y_x * distr
            self.q_y_x = q_y_x

        # Define the cost function
        if not config.use_logic:
            self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, dimension=1),tf.cast(self.y,tf.int64)), tf.float32))
            tf.summary.scalar('acc', self.acc)
            tf.summary.scalar('loss', self.total_cost)
            #self.auc_ROC = tf.metrics.auc(tf.cast(self.y,tf.int64), tf.arg_max(self.logits, dimension=1), curve = 'ROC')
            #self.auc_PR =  tf.metrics.auc(tf.cast(self.y,tf.int64), tf.arg_max(self.logits, dimension=1), curve = 'PR')
            #tf.summary.scalar('auc_ROC', self.auc_ROC)
            #tf.summary.scalar('auc_PR', self.auc_PR)
            # calculate acc 
        else:
            get_pi = lambda x, y: x * 0.9**tf.cast(y/6750, tf.float32)  # when batch size is 70: 6750; 48:9850
            pi = get_pi(config.pi, self.global_step)
            self.total_cost = (1-pi)*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
            self.total_cost += pi*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.arg_max(q_y_x, dimension=1), logits=self.logits))
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, dimension=1),tf.cast(self.y,tf.int64)), tf.float32))
            tf.summary.scalar('acc', self.acc)
            tf.summary.scalar('loss', self.total_cost)

        
        # L2 Loss
        if config.l2_loss:
            if config.sigmoid_growing_l2loss:
                weights_added = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() if tensor.name.endswith("weights:0") and not tensor.name.endswith("weighted_sum/weights:0") or tensor.name.endswith('kernel:0')])
                full_l2_step = tf.constant(config.weight_l2loss_step_full_reg , dtype=tf.int32, shape=[], name='full_l2reg_step')
                full_l2_ratio = tf.constant(config.l2_regularization_ratio , dtype=tf.float32, shape=[], name='l2_regularization_ratio')
                gs_flt = tf.cast(self.global_step , tf.float32)
                half_l2_step_flt = tf.cast(full_l2_step / 2 ,tf.float32)

                # (self.global_step - full_l2_step / 2)
                # tf.cast((self.global_step - full_l2_step / 2) * 8, tf.float32) / tf.cast(full_l2_step / 2 ,tf.float32)
                # l2loss_ratio = tf.sigmoid( tf.cast((self.global_step - full_l2_step / 2) * 8, tf.float32) / tf.cast(full_l2_step / 2 ,tf.float32)) * full_l2_ratio
                l2loss_ratio = tf.sigmoid( ((gs_flt - half_l2_step_flt) * 8) / half_l2_step_flt) * full_l2_ratio
                tf.summary.scalar('l2loss_ratio', l2loss_ratio)
                l2loss = weights_added * l2loss_ratio
            else:
                l2loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() if tensor.name.endswith("weights:0") or tensor.name.endswith('kernel:0')]) * tf.constant(config.l2_regularization_ratio , dtype='float', shape=[], name='l2_regularization_ratio')
            tf.summary.scalar('l2loss', l2loss)
            self.total_cost += l2loss

        # semantic Loss
        if config.semantic_loss:
            #semantic_loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() 
            #    if tensor.name.endswith("weights:0") or tensor.name.endswith('kernel:0')]) 
            
            def cal_exactly_one_loss(logits):
                #semantic_loss = tf.Variable(tf.zeros([], dtype=np.float32), name='semantic_loss_term')

                return tf.reduce_sum(-tf.log(logits[:,0]*(1-logits[:,1])*(1-logits[:,2]) +
                               logits[:,1]*(1-logits[:,0])*(1-logits[:,2]) +
                               logits[:,2]*(1-logits[:,0])*(1-logits[:,1])
                                ))

            def cal_logic_rules_loss(rules, logits):
                def cal_logic_rule(ro, ls):
                    if ro == '0':
                        return ls[0]*(1-ls[1])*(1-ls[2])
                    elif ro == '1':
                        return ls[1]*(1-ls[0])*(1-ls[2])
                    elif ro == '2':
                        return ls[2]*(1-ls[0])*(1-ls[1])
                return -tf.log(tf.add_n([cal_logic_rule(rule_output, logits) for rule_output in rules]))

            #semantic_loss = cal_logic_rules_loss(self.rules_output, self.logits)
            if config.use_exactly_one:
                semantic_loss = cal_exactly_one_loss(self.logits)
            semantic_loss = tf.reduce_mean(semantic_loss)
            semantic_loss = semantic_loss * tf.constant(config.semantic_regularization_ratio , dtype='float', shape=[], name='semantic_regularization_ratio')
            tf.summary.scalar('semantic loss', semantic_loss)
            self.total_cost += semantic_loss

        if config.wo_enc_sharing or config.wo_highway_sharing_but_penalize_diff:
            diffs = []
            for i in range(config.self_att_enc_layers):
                for tensor in tf.trainable_variables():
                    print(tensor.name)
                    if tensor.name == "prepro/{}_layer_self_att_enc/self_attention/h_logits/first/kernel:0".format(i):
                        l_lg = tensor 
                    elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_attention/h_logits/first/kernel:0".format(i):
                        r_lg = tensor 
                    elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/lhs_1/kernel:0".format(i):    
                        l_fg_lhs_1 = tensor 
                    elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/lhs_1/kernel:0".format(i):
                        r_fg_lhs_1= tensor
                    elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/rhs_1/kernel:0".format(i):
                        l_fg_rhs_1= tensor
                    elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/rhs_1/kernel:0".format(i):
                        r_fg_rhs_1= tensor
                    elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/lhs_2/kernel:0".format(i):
                        l_fg_lhs_2= tensor
                    elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/lhs_2/kernel:0".format(i):
                        r_fg_lhs_2= tensor
                    elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/rhs_2/kernel:0".format(i):
                        l_fg_rhs_2= tensor
                    elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/rhs_2/kernel:0".format(i):
                        r_fg_rhs_2= tensor

                    if config.two_gate_fuse_gate:
                        if tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/lhs_3/kernel:0".format(i):    
                            l_fg_lhs_3 = tensor 
                        elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/lhs_3/kernel:0".format(i):
                            r_fg_lhs_3 = tensor
                        elif tensor.name == "prepro/{}_layer_self_att_enc/self_att_fuse_gate/rhs_3/kernel:0".format(i):
                            l_fg_rhs_3 = tensor
                        elif tensor.name == "prepro/{}_layer_self_att_enc_h/self_att_fuse_gate/rhs_3/kernel:0".format(i):
                            r_fg_rhs_3 = tensor

                diffs += [l_lg - r_lg, l_fg_lhs_1 - r_fg_lhs_1, l_fg_rhs_1 - r_fg_rhs_1, l_fg_lhs_2 - r_fg_lhs_2, l_fg_rhs_2 - r_fg_rhs_2]
                if config.two_gate_fuse_gate:
                    diffs += [l_fg_lhs_3 - r_fg_lhs_3, l_fg_rhs_3 - r_fg_rhs_3]
            

            diff_loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in diffs]) * tf.constant(config.diff_penalty_loss_ratio , dtype='float', shape=[], name='diff_penalty_loss_ratio')
            tf.summary.scalar('diff_penalty_loss', diff_loss)
            self.total_cost += diff_loss


        self.summary = tf.summary.merge_all()

        total_parameters = 0
        for v in tf.global_variables():
            if not v.name.endswith("weights:0") and not v.name.endswith("biases:0") and not v.name.endswith('kernel:0') and not v.name.endswith('bias:0'):
                continue
            print(v.name)
            # print(type(v.name))
            shape = v.get_shape().as_list()
            param_num = 1
            for dim in shape:
                param_num *= dim 
            print(param_num)
            total_parameters += param_num
        print(total_parameters)


def bi_attention_mx(config, is_train, p, h, p_mask=None, h_mask=None, scope=None, wn_rel=None, sequence_length=48): #[N, L, 2d]
    with tf.variable_scope(scope or "dense_logit_bi_attention"):
        PL = p.get_shape().as_list()[1]
        if PL == None:
            PL = sequence_length
        HL = h.get_shape().as_list()[1]
        p_aug = tf.tile(tf.expand_dims(p, 2), [1,1,HL,1])
        h_aug = tf.tile(tf.expand_dims(h, 1), [1,PL,1,1]) #[N, PL, HL, 2d]

        if p_mask is None:
            ph_mask = None
        else:
            p_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, HL, 1]), tf.bool), axis=3)
            h_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(h_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
            ph_mask = p_mask_aug & h_mask_aug
        ph_mask = None

        
        h_logits = p_aug * h_aug  # [N, PL, HL, 2d]
        if config.use_more_interaction:
            h_logits_sub = p_aug - h_aug
            h_logits = tf.concat([h_logits, h_logits_sub], -1)   # [N, PL, HL, 2d+2d]

        if not config.concat_after_conv and wn_rel is not None:
            h_logits = tf.concat([h_logits, wn_rel], -1)   # [N, PL, HL, 2d+2d+5]
        return h_logits


def self_attention(config, is_train, p, p_mask=None, scope=None): #[N, L, 2d]
    with tf.variable_scope(scope or "self_attention"):
        PL = p.get_shape().as_list()[1]
        dim = p.get_shape().as_list()[-1]
        # HL = tf.shape(h)[1]
        p_aug_1 = tf.tile(tf.expand_dims(p, 2), [1,1,PL,1])
        p_aug_2 = tf.tile(tf.expand_dims(p, 1), [1,PL,1,1]) #[N, PL, HL, 2d]

        if p_mask is None:
            ph_mask = None
        else:
            p_mask_aug_1 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, PL, 1]), tf.bool), axis=3)
            p_mask_aug_2 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
            self_mask = p_mask_aug_1 & p_mask_aug_2

        h_logits = get_logits([p_aug_1, p_aug_2], None, True, wd=config.wd, mask=self_mask,
                              is_train=is_train, func=config.self_att_logit_func, scope='h_logits')  # [N, PL, HL]
        self_att = softsel(p_aug_2, h_logits) 

        return self_att


def self_attention_layer(config, is_train, p, p_mask=None, scope=None):
    with tf.variable_scope(scope or "self_attention_layer"):
        PL = tf.shape(p)[1]
        # HL = tf.shape(h)[1]
        # if config.q2c_att or config.c2q_att:
        self_att = self_attention(config, is_train, p, p_mask=p_mask)

        print("self_att shape")
        print(self_att.get_shape())
        
        p0 = fuse_gate(config, is_train, p, self_att, scope="self_att_fuse_gate")
        
        return p0


def get_denpendency(config, is_train, p, p_denp, p_mask=None, scope=None): #[N, L, 2d]
    """
    p: [70,48,448]
    p_denp: [70,48,48]
    """
    with tf.variable_scope(scope or "get_denpendency"):
        PL = p.get_shape().as_list()[1]     # 48
        dim = p.get_shape().as_list()[-1]   # 448
        # HL = tf.shape(h)[1]    
        p_wp = linear([p], config.dependency_hidden_size, True, scope='get_denpendency_wp', wd=config.wd, is_train=is_train)  # 70*48*448
        ci_1 = tf.tile(tf.expand_dims(p_denp, 3), [1,1,1,config.dependency_hidden_size]) # 70*48*48*448
        ci_2 = tf.tile(tf.expand_dims(p, 2), [1,1,PL,1])
        ci = tf.to_float(ci_1) * ci_2
        cis = tf.reduce_sum(ci, axis=2)  # 70*48*448
        c_wc = linear([cis], config.dependency_hidden_size, True, scope='get_denpendency_wc', wd=config.wd, is_train=is_train)  # 70*48*448

        logits = p_wp + c_wc
        if p_mask is not None:
            logits = exp_mask(logits, p_mask)
        logits = tf.nn.relu(logits)

        return logits

def to_one_hot(t):
    """
    t: 70*48*6
    return: 70*48*48
    """
    PL = t.get_shape().as_list()[1]
    one_hots = tf.one_hot(t, depth = PL)  # 70*48*6*48
    return tf.reduce_sum(one_hots, axis=2)

def get_denpendency1(config, is_train, p, p_denp, p_mask=None, scope=None): #[N, L, 2d]
    """
    p: [70,48,448]
    p_denp: [70,48,6]
    """
    with tf.variable_scope(scope or "get_denpendency"):
        PL = p.get_shape().as_list()[1]     # 48
        dim = p.get_shape().as_list()[-1]   # 448
        # HL = tf.shape(h)[1]    
        p_denp = to_one_hot(p_denp)
        p_wp = linear([p], config.dependency_hidden_size, True, scope='get_denpendency_wp', wd=config.wd, is_train=is_train)  # 70*48*448
        ci_1 = tf.tile(tf.expand_dims(p_denp, 3), [1,1,1,dim]) # 70*48*48*448
        ci_2 = tf.tile(tf.expand_dims(p, 2), [1,1,PL,1])
        ci = tf.to_float(ci_1) * ci_2
        cis = tf.reduce_sum(ci, axis=2)  # 70*48*448
        c_wc = linear([cis], config.dependency_hidden_size, True, scope='get_denpendency_wc', wd=config.wd, is_train=is_train)  # 70*48*448

        logits = p_wp + c_wc
        if p_mask is not None:
            logits = exp_mask(logits, p_mask)
        logits = tf.nn.relu(logits)

        return logits


def dependency_layer(config, is_train, p, p_denp, p_mask=None, scope=None):
    with tf.variable_scope(scope or "dependency_layer"):
        PL = tf.shape(p)[1]
        # HL = tf.shape(h)[1]
        # if config.q2c_att or config.c2q_att:
        denp = get_denpendency1(config, is_train, p, p_denp, p_mask=p_mask)

        print("dependency shape")
        print(denp.get_shape())
        
        p0 = denp
        #p0 = fuse_gate(config, is_train, p, denp, scope="dependency_fuse_gate")
        
        return p0



# def bi_attention(config, is_train, p, h, p_mask=None, h_mask=None, scope=None, h_value = None): #[N, L, 2d]
#     with tf.variable_scope(scope or "bi_attention"):
#         PL = tf.shape(p)[1]
#         HL = tf.shape(h)[1]
#         p_aug = tf.tile(tf.expand_dims(p, 2), [1,1,HL,1])
#         h_aug = tf.tile(tf.expand_dims(h, 1), [1,PL,1,1]) #[N, PL, HL, 2d]


#         if p_mask is None:
#             ph_mask = None
#         else:
#             p_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, HL, 1]), tf.bool), axis=3)
#             h_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(h_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
#             ph_mask = p_mask_aug & h_mask_aug


#         h_logits = get_logits([p_aug, h_aug], None, True, wd=config.wd, mask=ph_mask,
#                           is_train=is_train, func="mul_linear", scope='h_logits')  # [N, PL, HL]
#         h_a = softsel(h_aug, h_logits) 
#         p_a = softsel(p, tf.reduce_max(h_logits, 2))  # [N, 2d]
#         p_a = tf.tile(tf.expand_dims(p_a, 1), [1, PL, 1]) # 

#         return h_a, p_a


def dense_net(config, denseAttention, is_train, wn_rel=None):
    with tf.variable_scope("dense_net"):
        
        dim = denseAttention.get_shape().as_list()[-1]
        print('denset net dim: %s' % dim)
        act = tf.nn.relu if config.first_scale_down_layer_relu else None
        fm = tf.contrib.layers.convolution2d(denseAttention, int(dim * config.dense_net_first_scale_down_ratio), config.first_scale_down_kernel, padding="SAME", activation_fn = act)

        if config.concat_after_conv and wn_rel is not None:
           fm = tf.concat([fm, wn_rel], -1)   # [N, PL, HL, 2d*scale_down_ratio+5]
        
        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers, config.dense_net_kernel_size, is_train ,scope = "first_dense_net_block") 
        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='second_transition_layer')
        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers, config.dense_net_kernel_size, is_train ,scope = "second_dense_net_block") 
        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='third_transition_layer')
        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers, config.dense_net_kernel_size, is_train ,scope = "third_dense_net_block") 

        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='fourth_transition_layer')

        shape_list = fm.get_shape().as_list()
        print(shape_list)
        out_final = tf.reshape(fm, [-1, shape_list[1]*shape_list[2]*shape_list[3]])
        return out_final



def dense_net_block(config, feature_map, growth_rate, layers, kernel_size, is_train, padding="SAME", act=tf.nn.relu, scope=None):
    with tf.variable_scope(scope or "dense_net_block"):
        conv2d = tf.contrib.layers.convolution2d
        dim = feature_map.get_shape().as_list()[-1]

        list_of_features = [feature_map]
        features = feature_map
        for i in range(layers):        
            ft = conv2d(features, growth_rate, (kernel_size, kernel_size), padding=padding, activation_fn=act)
            list_of_features.append(ft)
            features = tf.concat(list_of_features, axis=3)

        print("dense net block out shape")
        print(features.get_shape().as_list())
        return features 

def dense_net_transition_layer(config, feature_map, transition_rate, scope=None):
    with tf.variable_scope(scope or "transition_layer"):


        out_dim = int(feature_map.get_shape().as_list()[-1] * transition_rate)
        feature_map = tf.contrib.layers.convolution2d(feature_map, out_dim, 1, padding="SAME", activation_fn = None)
        
        
        feature_map = tf.nn.max_pool(feature_map, [1,2,2,1],[1,2,2,1], "VALID")

        print("Transition Layer out shape")
        print(feature_map.get_shape().as_list())
        return feature_map


