import maze from '@indutny/maze';

export default class Environment {
  static ACTION_DIMS = 4;
  static OBSERVATION_DIMS = 0;

  constructor() {
    this.reset();
  }

  reset() {
    this.hits = 0;
  }

  observe() {
    return [];
  }

  actionMask() {
    return [ 1, 1, 1, 1 ];
  }

  step(action) {
    let dice = false;
    if (action === 0) {
      dice = Math.random() > 0.75;
    } else if (action === 1) {
      dice = Math.random() > 0.5;
    } else if (action === 2) {
      dice = Math.random() > 0.25;
    } else if (action === 3) {
      dice = Math.random() > 0.01;
    }

    if (dice) {
      this.hits++;
      return 1;
    } else {
      return 0;
    }
  }

  isFinished() {
    return false;
  }

  printTrace() {
  }
}
