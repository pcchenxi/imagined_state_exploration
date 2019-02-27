import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn

# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            is_training = tf.placeholder_with_default(False, shape=(), name='training')

            # extracted_features = nature_cnn(self.processed_obs, **kwargs)
            # extracted_features = tf.layers.flatten(extracted_features)

            pi_h = tf.layers.flatten(self.processed_obs)
            vf_h = tf.layers.flatten(self.processed_obs)
            rp_h = tf.layers.flatten(self.processed_obs)

            with tf.variable_scope('ac_net'):
                for i, layer_size in enumerate([64, 32]):
                    pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
                pi_latent = pi_h

                for i, layer_size in enumerate([64, 32]):
                    vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
                vf_latent = vf_h
                value_fn = tf.layers.dense(vf_h, 1, name='vf')

                self.proba_distribution, self.policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)


            with tf.variable_scope('reward_class'):
                for i, layer_size in enumerate([1024, 64, 32]):
                    rp_h = activ(tf.layers.dense(rp_h, layer_size, name='rp_fc' + str(i)))
                    # drop = tf.layers.dropout(inputs=rp_h, rate=0.5, training=is_training)
                rp_logits = tf.layers.dense(rp_h, 2, name='r_p')
                
                rp_prob = tf.nn.softmax(rp_logits, name="softmax_tensor")
                rp_class = tf.argmax(input=rp_logits, axis=1)

        self.value_fn = value_fn
        self.rp_logits = rp_logits 
        self.rp_prob = rp_prob 
        self.rp_class = rp_class 

        self.initial_state = None
        self._setup_init()

        self.is_training = is_training

    def step(self, obs, state=None, mask=None, deterministic=False):
        action, value, reward, neglogp = self.sess.run([self.action, self._value, self.rp_prob, self.neglogp], {self.obs_ph: obs})
        # return action, value, reward[:,-1], self.initial_state, neglogp
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})

    def pred_reward(self, obs):
        prob, label = self.sess.run([self.rp_prob, self.rp_class], {self.obs_ph: obs})
        return prob[:, -1], label