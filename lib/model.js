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
      activation: 'softmax',
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

  forward(observation, state) {
    const newState = this.lstm.call([ observation ].concat(state), {
      training: true,
    });

    const lstmOut = newState.shift();
    const probs = this.probs.call(lstmOut);

    return { probs, newState };
  }

  sample(probs) {
    return tf.multinomial(probs, 1);
  }
}
