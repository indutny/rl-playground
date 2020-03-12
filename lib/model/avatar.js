import createDebug from 'debug';

import Model from './base.js';

const debug = createDebug('rl:avatar');

// NOTE: Useful for self-play
export default class Avatar extends Model {
  constructor(Environment, Model, { source, generationOff, generationMod }) {
    super('avatar', Environment);

    this.Model = Model;

    this.inner = new Model(Environment);

    this.source = source;
    this.generationOff = generationOff;
    this.generationMod = generationMod;

    if (typeof this.generationOff !== 'number') {
      throw new Error('`generationOff` must be a number');
    }
    if (typeof this.generationMod !== 'number') {
      throw new Error('`generationMod` must be a number');
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
    return await super.step(args);
  }

  evolve() {
    if (this.source.generation % this.generationMod === this.generationOff) {
      debug(`copying source#${this.source.generation} to ` +
        `avatar#${this.generation}`);
      this.source.copy(this.inner);
      this.generation = this.source.generation;
    }
  }
}
