import tf from '@tensorflow/tfjs-node';

class Tower {
  constructor(name, options) {
    const {
      lstm,
      hidden = [],
      lastActivation = undefined,
    } = options;

    this.lstm = tf.layers.lstmCell({
      name: `${name}_lstm`,
      units: lstm,
    });
    this.hidden = [];

    for (const [ i, units ] of hidden.entries()) {
      const isLast = i === hidden.length - 1;

      const l = tf.layers.dense({
        name: `${name}_i`,
        units,
        activation: isLast ? lastActivation : 'relu',
      });

      this.hidden.push(l);
    }

    this.initialState = null;
  }

  build(inputShape) {
    const batchSize = inputShape[0];
    this.lstm.build(inputShape);

    inputShape = [ batchSize, this.lstm.units ];
    for (const l of this.hidden) {
      l.build(inputShape);
      inputShape = l.computeOutputShape(inputShape);
    }

    this.initialState = this.lstm.stateSize.map((size) => {
      return tf.zeros([ batchSize, size ]);
    });
  }

  forward(observation, state, training = false) {
    return tf.tidy(() => {
      const newState = this.lstm.call([ observation ].concat(state), {
        training,
      });

      let out = newState.shift();
      for (const l of this.hidden) {
        out = l.call(out);
      }

      return { out, newState };
    });
  }

  getWeights() {
    let out = [];
    out = out.concat(this.lstm.getWeights());
    for (const l of this.hidden) {
      out = out.concat(l.getWeights());
    }
    return out;
  }
}

export default class Model {
  constructor(Environment) {
    this.probs = new Tower('probs', {
      lstm: 16,
      hidden: [ Environment.ACTION_DIMS ],
    });

    this.value = new Tower('value', {
      lstm: 8,
      hidden: [ 1 ],
    });

    this.initialState = null;
  }

  build(inputShape) {
    this.probs.build(inputShape);
    this.value.build(inputShape);

    this.initialState = {
      probs: this.probs.initialState,
      value: this.value.initialState,
    };
  }

  getWeights() {
    let out = [];
    out = out.concat(this.probs.getWeights());
    out = out.concat(this.value.getWeights());
    return out;
  }

  forward(observation, state, actionMask) {
    return tf.tidy(() => {
      const { out: probsOut, newState: probsNewState } = this.probs.forward(
        observation, state.probs);

      const probs = tf.logSoftmax(probsOut.mul(actionMask)
        .add(tf.tensor(1).sub(actionMask).mul(tf.tensor(-1e23))));

      const { out: value, newState: valueNewState } = this.value.forward(
        observation, state.value);

      return {
        probs,
        value: value.squeeze(-1),
        newState: { probs: probsNewState, value: valueNewState },
      };
    });
  }

  sample(observation, state, actionMask) {
    return tf.tidy(() => {
      const {
        probs,
        value,
        newState,
      } = this.forward(observation, state, actionMask);

      return {
        action: tf.multinomial(probs, 1).squeeze(-1),
        value,
        newState,
      };
    });
  }
}
