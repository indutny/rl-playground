import tf from '@tensorflow/tfjs-node';

import Model from './base.js';

class Tower {
  constructor(name, options) {
    const {
      before = [],
      lstm,
      after = [],
      activation = 'selu',
      lastActivation = undefined,
    } = options;

    this.lstm = lstm && tf.layers.lstmCell({
      name: `${name}_lstm`,
      units: lstm,
    });

    this.before = [];
    this.after = [];

    for (const [ i, units ] of before.entries()) {
      const l = tf.layers.dense({
        name: `${name}_before_${i}`,
        units,
        activation,
      });

      this.before.push(l);
    }

    for (const [ i, units ] of after.entries()) {
      const isLast = i === after.length - 1;

      const l = tf.layers.dense({
        name: `${name}_after_${i}`,
        units,
        activation: isLast ? lastActivation : activation,
      });

      this.after.push(l);
    }

    this.initialState = null;
  }

  build(inputShape) {
    for (const l of this.before) {
      l.build(inputShape);
      inputShape = l.computeOutputShape(inputShape);
    }

    if (this.lstm) {
      this.lstm.build(inputShape);
      inputShape = [ this.lstm.units ];
    }

    for (const l of this.after) {
      l.build(inputShape);
      inputShape = l.computeOutputShape(inputShape);
    }

    this.initialState = this.lstm && this.lstm.stateSize.map((size) => {
      return tf.zeros([ 1, size ]);
    });
  }

  forward(observation, state, training = false) {
    return tf.tidy(() => {
      let out = observation;

      for (const l of this.before) {
        out = l.call(out);
      }

      let newState = null;
      if (this.lstm) {
        out = [ out ].concat(state);

        newState = this.lstm.call(out, {
          training,
        });
        out = newState.shift();
      }

      for (const l of this.after) {
        out = l.call(out);
      }

      return { out, newState };
    });
  }

  getWeights() {
    let out = [];
    if (this.lstm) {
      out = out.concat(this.lstm.getWeights());
    }
    for (const l of this.after) {
      out = out.concat(l.getWeights());
    }
    return out;
  }

  copy(to) {
    const weights = this.getWeights();
    const toWeights = to.getWeights();

    if (weights.length !== toWeights.length) {
      throw new Error('Either of models is not built yet!');
    }

    for (const [ i, v ] of weights.entries()) {
      toWeights[i].assign(v);
    }
  }
}

export default class RNNModel extends Model {
  constructor(Environment) {
    super('rnn', Environment);

    this.actor = new Tower('actor', {
      lstm: 64,
      after: [ Environment.ACTION_DIMS ],
    });
    this.critic = new Tower('critic', {
      lstm: 16,
      after: [ 1 ],
    });
  }

  build(inputShape) {
    this.actor.build(inputShape);
    this.critic.build(inputShape);

    this.initialState = {
      actor: this.actor.initialState,
      critic: this.critic.initialState,
    };
  }

  getWeights() {
    let out = [];
    out = out.concat(this.actor.getWeights());
    out = out.concat(this.critic.getWeights());
    return out;
  }

  copy(to) {
    this.actor.copy(to.actor);
    this.critic.copy(to.critic);
  }

  forward(observation, state, actionMask) {
    return tf.tidy(() => {
      let { out: probs, newState: newActorState } = this.actor.forward(
        observation, state.actor);
      const { out: value, newState: newCriticState } = this.critic.forward(
        observation, state.critic);

      const negativeMask = tf.tensor(1).sub(actionMask);

      // Leave only valid actions
      probs = probs.mul(actionMask);

      // Send invalid actions to -Infinity
      probs = probs.add(negativeMask.mul(tf.tensor(-1e23)));

      // Apply `logSoftmax` that `tf.multinomial` expects
      const logProbs = tf.logSoftmax(probs);

      return {
        logProbs,
        value: value.squeeze(-1),
        newState: { actor: newActorState, critic: newCriticState },
      };
    });
  }
}
