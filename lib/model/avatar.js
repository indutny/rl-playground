import tf from '@tensorflow/tfjs-node';

import Model from './base.js';

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
    // Time to copy the weights
    if (Math.random() >= 1 - this.copyProbability) {
      this.source.inner.copy(this.inner);
      this.generation = this.source.generation;
    }

    return await this.inner.step(args);
  }
}
