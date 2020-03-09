import tf from '@tensorflow/tfjs-node';

export default class Model {
  constructor(Environment) {
    this.lstm = tf.layers.lstmCell({
      name: 'lstm',
      units: 16,
    });
    this.probs = tf.layers.dense({
      name: 'probs',
      units: Environment.ACTION_DIMS,
    });

    this.value = [
      tf.layers.dense({ units: 16, activation: 'relu' }),
      tf.layers.dense({ units: 8, activation: 'relu' }),
      tf.layers.dense({ units: 1, activation: 'relu' }),
    ];

    this.initialState = null;
  }

  build(inputShape) {
    const batchSize = inputShape[0];
    this.lstm.build(inputShape);

    const lstmShape = [ batchSize, this.lstm.units ];
    this.probs.build(lstmShape);

    this.buildValue(lstmShape);

    this.initialState = this.lstm.stateSize.map((size) => {
      return tf.zeros([ inputShape[0], size ]);
    });
  }

  buildValue(inputShape) {
    for (const l of this.value) {
      l.build(inputShape);
      inputShape = l.computeOutputShape(inputShape);
    }
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

      let value = lstmOut;
      for (const l of this.value) {
        value = l.call(value);
      }
      value = value.squeeze(-1);

      return { probs, value, newState };
    });
  }

  sample(observation, state, actionMask) {
    return tf.tidy(() => {
      const {
        probs,
        newState,
        value,
      } = this.forward(observation, state, actionMask);

      return {
        action: tf.multinomial(probs, 1).squeeze(-1),
        newState,
        value,
      };
    });
  }
}
