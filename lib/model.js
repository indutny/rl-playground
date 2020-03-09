import tf from '@tensorflow/tfjs-node';

export default class Model {
  constructor(Environment) {
    this.lstm = tf.layers.lstmCell({
      name: 'lstm',
      units: 32,
    });
    this.probs = tf.layers.dense({
      name: 'probs',
      units: Environment.ACTION_DIMS,
    });

    this.initialState = null;
  }

  build(inputShape) {
    const batchSize = inputShape[0];
    this.lstm.build(inputShape);
    this.probs.build([ batchSize, this.lstm.units ]);

    this.initialState = this.lstm.stateSize.map((size) => {
      return tf.zeros([ inputShape[0], size ]);
    });
  }

  forward(observation, state, actionMask) {
    return tf.tidy(() => {
      const newState = this.lstm.call([ observation ].concat(state), {
        training: true,
      });

      const lstmOut = newState.shift();
      const probs = tf.logSoftmax(
        this.probs.call(lstmOut).mul(actionMask)
          .add(tf.tensor(1).sub(actionMask).mul(tf.tensor(-1e23))));

      return { probs, newState };
    });
  }

  sample(observation, state, actionMask) {
    return tf.tidy(() => {
      const { probs, newState } = this.forward(observation, state, actionMask);
      return {
        action: tf.multinomial(probs, 1).squeeze(-1),
        newState,
      };
    });
  }
}
