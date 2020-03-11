import tf from '@tensorflow/tfjs-node';

export default class Model {
  constructor(name, Environment) {
    this.name = name;
    this.Environment = Environment;
    this.generation = 0;

    this.initialState = null;
  }

  build(inputShape) {
  }

  getWeights() {
    return [];
  }

  copy(to) {
  }

  // Called during training
  evolve() {
    this.generation++;
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

  async step(options) {
    const {
      envs,
      state: stateTensors,
      history = undefined,
    } = options;

    const observation = tf.tensor(await Promise.all(
      envs.map((env) => env.observe())));

    const actionMask = tf.tensor(envs.map((env) => env.actionMask()));
    const {
      action: actionTensor,
      newState: newStateTensor,
      value: valueTensor,
    } = this.sample(observation, stateTensors, actionMask);

    const [ action, value ] = await Promise.all([
      actionTensor.data(),
      valueTensor.data(),
    ]);

    tf.dispose(actionTensor);
    tf.dispose(valueTensor);

    // Update environment
    const rewards = new Array(envs.length);
    for (const [ i, env ] of envs.entries()) {
      const reward = await env.step(action[i]);
      if (rewards) {
        rewards[i] = reward;
      }
    }

    const historyEntry = {
      // These two are tensors:
      observation,
      actionMask,
      state: stateTensors === this.initialState ? undefined : stateTensors,

      // The rest are not tensors
      action,
      rewards,
      value,
    };

    if (history) {
      history.push(historyEntry);
    } else {
      tf.dispose(historyEntry);
    }

    return newStateTensor;
  }
}
