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

  async step({ envs, state, history, rewards }) {
    const observation =
      tf.tensor(await Promise.all(envs.map((env) => env.observe())));

    const actionMask = tf.tensor(envs.map((env) => env.actionMask()));
    const {
      action: actionTensor,
      newState,
      value: valueTensor,
    } = this.sample(observation, state, actionMask);

    const [ action, value ] = await Promise.all([
      actionTensor.data(),
      valueTensor.data(),
    ]);
    tf.dispose(actionTensor);
    tf.dispose(valueTensor);

    history.push({
      observation,
      actionMask,
      state: state === this.initialState ? undefined : state,
      action,
      rewards: rewards.slice(),
      value,
    });

    // Update environment
    let allFinished = true;
    for (const [ i, env ] of envs.entries()) {
      rewards[i] += await env.step(action[i]);
      if (!env.isFinished()) {
        allFinished = false;
      }
    }

    if (allFinished) {
      tf.dispose(newState);
      return null;
    }

    return newState;
  }

  async loop(envs, { maxSteps }) {
    let state = this.initialState;
    const rewards = new Array(envs.length).fill(0);

    const history = [];
    for (let t = 0; t < maxSteps; t++) {
      state = await this.step({
        envs,
        state,
        history,
        rewards,
      });

      // All finished!
      if (!state) {
        break;
      }
    }
    if (state !== this.initialState) {
      tf.dispose(state);
    }

    return { finalRewards: rewards, history };
  }
}
