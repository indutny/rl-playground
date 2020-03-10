import Environment from './base.js';

const SEQUENCE = [ 1, 3, 1, 6, 1, 8, 3, 0, 6, 3, 7, 9, 7, 8, 1, 5, 6 ];

export default class Sequence extends Environment {
  static ACTION_DIMS = 10;
  static OBSERVATION_DIMS = SEQUENCE.length + 1;

  constructor() {
    super('sequence');
    this.reset();
  }

  reset() {
    this.offset = 0;
  }

  observe() {
    const out = new Array(SEQUENCE.length + 1).fill(0);
    out[this.offset] = 1;
    return out;
  }

  actionMask() {
    return new Array(Environment.ACTION_DIMS).fill(1);
  }

  step(action) {
    if (this.offset === SEQUENCE.length) {
      return 0;
    }

    let reward = 0;
    if (SEQUENCE[this.offset] === action) {
      this.offset++;
      reward = 1;
    } else {
      this.offset = SEQUENCE.length;
      reward = 0;
    }

    return reward;
  }

  isFinished() {
    return this.offset === SEQUENCE.length;
  }
}
