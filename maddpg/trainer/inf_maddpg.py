import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U
import maddpg.trainer

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer
from maddpg.trainer.maddpg import MADDPGAgentTrainer, make_update_exp

import tensorflow.contrib.layers as layers
    
def i_train(make_obs_ph_n, act_space_n, j_index, inf_func, optimizer,
            grad_norm_clipping=None, num_units=64, scope="inference", reuse=None):
    """ u_i^j is agent i's approximation of agent j's policy

    It is mapping between agent j's observation and (the probability of) 
      agent j's actions.
    
    It is trained by maximizing the log probability of agent j's actions, with 
      an entropy regularizer.
    """

    with tf.variable_scope(scope, reuse=reuse):
        # Create distribution (should be a soft categorical)
        act_pdtype_j = make_pdtype(act_space_n[j_index])

        # Create placeholders
        obs_ph_j = make_obs_ph_n[j_index]
        act_ph_j = act_pdtype_j.sample_placeholder([None], name="action"+str(j_index))
        target_ph_j = act_pdtype_j.sample_placeholder([None], name="target"+str(j_index))

        inf_output = int(act_pdtype_j.param_shape()[0])

        # i for inference
        i_j = inf_func(obs_ph_j, inf_output, scope="i_func_"+str(j_index), num_units=num_units)
        i_func_vars = U.scope_vars(U.absolute_scope_name("i_func_"+str(j_index)))

        i_loss = tf.losses.softmax_cross_entropy(target_ph_j, i_j)

        act_pd = act_pdtype_j.pdfromflat(i_j)
        h = act_pd.entropy()
        loss = i_loss + h
        
        optimize_expr = U.minimize_and_clip(optimizer, loss, i_func_vars, grad_norm_clipping)

        # Create callable functions
        i_act = U.function(inputs=[obs_ph_j], outputs=[act_ph_j])
        i_values = U.function([obs_ph_j], i_j)

        
        target_i = inf_func(obs_ph_j, inf_output, scope="target_i_func_"+str(j_index),
                            num_units=num_units)
        target_i_func_vars = U.scope_vars(U.absolute_scope_name("target_i_func_"+str(j_index)))
        update_target_i = make_update_exp(i_func_vars, target_i_func_vars)
        target_i_act = U.function([obs_ph_j], target_i)
        
        i_train = U.function(inputs=[obs_ph_j, target_ph_j], outputs=loss, updates=[optimize_expr])
        return i_act, i_train, update_target_i, {'i_values' : i_values,
                                                 'target_act' : target_i_act}

class InferenceMADDPGAgentTrainer(MADDPGAgentTrainer):
    """ MTL MADDPG AgentTrainer
    Extension of MADDPG to make use of inference.

    Each agent does not know other agent's policies, but instead has a model
    of other agent's policies. The other agent's actions are known.
    """

    def __init__(self, name, model, obs_shape_n, act_shape_n, agent_index, args, local_q_func=False):
        super().__init__(name, model, obs_shape_n, act_shape_n, agent_index, args, False)

        self.inf_acts = []
        self.inf_trains = []
        self.inf_updates  = []
        self.inf_debugs   = []

        obs_ph_n = self.obs_ph_n
        
        for i in range(self.n):
            if not i == self.agent_index:
                inf_act, inf_train, inf_update, inf_debug = i_train(
                    scope=self.name,
                    make_obs_ph_n=obs_ph_n,
                    act_space_n=act_shape_n,
                    inf_func=model,
                    optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
                    grad_norm_clipping=0.5,
                    num_units=args.num_units,
                    j_index=i
                )
            else:
                inf_act = None #self.p_debug
                inf_train = None
                inf_update = None
                inf_debug = self.p_debug

            self.inf_acts.append(inf_act)
            self.inf_trains.append(inf_train)
            self.inf_updates.append(inf_update)
            self.inf_debugs.append(inf_debug)

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            return
        if not t % 100 == 0:
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)

        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index

        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        target_act_next_n = []
        
        for i in range(self.n):
            obs_next_i = obs_next_n[i]
            inf_debug_i = self.inf_debugs[i]
            act_f_i = inf_debug_i['target_act']
            target_act_next_i = act_f_i(obs_next_i)
            target_act_next_n.append(target_act_next_i)
#            target_act_next_n.append(self.inf_debugs[i]['target_act'](*(obs_next_n[i])))
            
        target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
        target_q = rew + self.args.gamma * (1.0 - done) * target_q_next

        q_loss = self.q_train(*(obs_n + act_n + [target_q]))
        p_loss = self.p_train(*(obs_n + act_n))
        
        for i in range(self.n):
            if not i == self.agent_index:
                i_loss = self.inf_trains[i](*([obs_n[i]] + [act_n[i]]))
                self.inf_updates[i]()

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
                
            

        

        
