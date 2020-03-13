import Environment from './base.js';

const SEQUENCE = [
  3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3, 3,
  8, 3, 2, 7, 9, 5, 0, 2, 8, 8, 4, 1, 9, 7, 1, 6, 9, 3, 9, 9, 3, 7, 5, 1, 0, 5,
  8, 2, 0, 9, 7, 4, 9, 4, 4, 5, 9, 2, 3, 0, 7, 8, 1, 6, 4, 0, 6, 2, 8, 6, 2, 0,
  8, 9, 9, 8, 6, 2, 8, 0, 3, 4, 8, 2, 5, 3, 4, 2, 1, 1, 7, 0, 6, 7, 9,
];

export default class Sequence extends Environment {
  static ACTION_DIMS = 10;
  static OBSERVATION_DIMS = 0;
  static PLAYER_COUNT = 1;

  constructor() {
    super('sequence');
    this.reset();
  }

  reset() {
    this.offset = 0;
  }

  async observe() {
    const out = new Array(Sequence.OBSERVATION_DIMS).fill(0);
    if (out.length !== 0) {
      out[this.offset % out.length] = 1;
    }
    return out;
  }

  actionMask() {
    return new Array(Sequence.ACTION_DIMS).fill(1);
  }

  async step(action) {
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
