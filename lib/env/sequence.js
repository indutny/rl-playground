import maze from '@indutny/maze';

const SEQUENCE = [ 1, 3, 1, 6, 1, 8, 3, 0, 6, 3, 7, 9, 7, 8, 1, 5, 6 ];

export default class Environment {
  static ACTION_DIMS = 10;
  static OBSERVATION_DIMS = 1;

  constructor() {
    this.reset();
  }

  reset() {
    this.offset = 0;
    this.hits = 0;
  }

  observe() {
    return [ this.offset / SEQUENCE.length ];
  }

  step(action) {
    let reward = 0;
    if (SEQUENCE[this.offset++] === action) {
      this.hits++;
      reward = 1 / SEQUENCE.length;
    }
    if (this.offset === SEQUENCE.length) {
      this.offset = 0;
    }

    return reward;
  }

  isFinished() {
    return false;
  }

  printTrace() {
  }

  get score() {
    return this.hits;
  }
}
