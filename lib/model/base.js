import tf from '@tensorflow/tfjs-node';

export default class Model {
  constructor(name, Environment) {
    this.name = name;
    this.Environment = Environment;

    this.initialState = null;
  }

  build(inputShape) {
  }

  getWeights() {
    return [];
  }

  copy(to) {
  }

  forward(observation, state, actionMask) {
    return tf.tensor(0);
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
