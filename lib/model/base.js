import { Buffer } from 'buffer';
import { promisify } from 'util';
import zlib from 'zlib';

import tf from '@tensorflow/tfjs-node';

export default class Model {
  constructor(name, Environment) {
    this.name = name;
    this.Environment = Environment;
    this.generation = 0;

    this.initialState = null;
  }

  build() {
  }

  async toJSON() {
    const out = {};

    for (const v of this.getWeights()) {
      const buffer = Buffer.from(await v.bytes());
      const compressed = await promisify(zlib.deflate)(buffer);

      if (out[v.name]) {
        throw new Error(`Duplicate variable: "${v.name}"`);
      }

      out[v.name] = {
        shape: v.shape,
        compressed: compressed.toString('base64'),
      };
    }

    return JSON.stringify(out);
  }

  async loadJSON(json) {
    const list = JSON.parse(json);

    for (const v of this.getWeights()) {
      if (!list[v.name]) {
        throw new Error(`Variable "${v.name}" not found`);
      }

      const entry = list[v.name];
      const compressed = Buffer.from(entry.compressed, 'base64');
      const buffer = await promisify(zlib.inflate)(compressed);

      const bytes = await v.bytes();
      if (bytes.length !== buffer.length) {
        throw new Error('Invalid variable shape!');
      }

      for (let i = 0; i < bytes.length; i++) {
        bytes[i] = buffer[i];
      }
    }
  }

  getWeights() {
    return [];
  }

  copy() {
  }

  // Called during training
  evolve() {
    this.generation++;
  }

  forward() {
    const zero = tf.tensor(0);
    return { logProbs: zero, value: zero, newState: zero };
  }

  sample(observation, state, actionMask) {
    return tf.tidy(() => {
      const {
        logProbs,
        value,
        newState,
      } = this.forward(observation, state, actionMask);

      const action = tf.multinomial(logProbs, 1).squeeze(-1);
      const sampleMask = tf.oneHot(action, this.Environment.ACTION_DIMS);

      return {
        action,
        sampleMask,
        value,
        newState,
        logProbs,
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
      sampleMask: sampleMaskTensor,
      newState: newStateTensor,
      value: valueTensor,
      logProbs: logProbsTensor,
    } = this.sample(observation, stateTensors || this.initialState, actionMask);

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
      state: stateTensors,
      logProbs: logProbsTensor,
      sampleMask: sampleMaskTensor,

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
