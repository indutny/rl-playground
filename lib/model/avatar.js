import tf from '@tensorflow/tfjs-node';
import createDebug from 'debug';

import Model from './base.js';

const debug = createDebug('rl:avatar');

// NOTE: Useful for self-play
export default class Avatar extends Model {
  constructor(Environment, Model, { source, copyProbability }) {
    super('avatar', Environment);

    this.Model = Model;

    this.inner = new Model(Environment);

    this.source = source;
    this.copyProbability = copyProbability;

    if (typeof this.copyProbability !== 'number') {
      throw new Error('`copyProbability` must be a number');
    }
  }

  build(inputShape) {
    this.inner.build(inputShape);
    this.initialState = this.inner.initialState;
  }

  getWeights() {
    return this.inner.getWeights();
  }

  copy(to) {
    this.inner.copy(to.inner);
  }

  forward(observation, state, actionMask) {
    return this.inner.forward(observation, state, actionMask);
  }

  async step(args) {
    const { state } = args;
    // Time to copy the weights
    if (state === this.initialState &&
        Math.random() >= 1 - this.copyProbability) {
      debug(`copying source#${this.source.generation} to ` +
        `avatar#${this.generation}`);
      this.source.copy(this.inner);
      this.generation = this.source.generation;
    }

    return await super.step(args);
  }
}
