import Environment from './base.js';

export default class TicTacToe extends Environment {
  static ACTION_DIMS = 3;
  static OBSERVATION_DIMS = 2;

  constructor() {
    super('prisoner');

    this.reset();
  }

  reset() {
    this.lastStack = [ -1, -1 ];
    this.stack = [];
    this.stage = 0;
    this.trace = [];
  }

  async observe() {
    if (this.stage % 2 === 0) {
      return this.lastStack;
    } else {
      return [ this.lastStack[1], this.lastStack[0] ];
    }
  }

  actionMask() {
    // First they vote, then they get the reward
    if (this.stage < 2) {
      return [ 1, 1, 0 ];
    } else {
      return [ 0, 0, 1 ];
    }
  }

  async step(action) {
    let reward = 0;
    if (this.stage < 2) {
      this.stack.push(action);
    } else {
      this.trace.push(this.stack);

      if (this.stack[0] === 1 && this.stack[1] === 1) {
        reward = 0.5;
      } else if (this.stack[0] === 0 && this.stack[1] === 1) {
        reward = this.stage === 2 ? 0.6 : 0;
      } else if (this.stack[1] === 0 && this.stack[0] === 1) {
        reward = this.stage === 3 ? 0.6 : 0;
      } else if (this.stack[0] === 0 && this.stack[1] === 0) {
        reward = 0;
      }

      if (this.stage === 3) {
        this.lastStack = this.stack;
        this.stack = [];
      }
    }
    this.stage = (this.stage + 1) % 4;
    return reward;
  }

  toString() {
    return this.trace.map((stack) => {
      return stack.join('-');
    }).join('\n');
  }
}
