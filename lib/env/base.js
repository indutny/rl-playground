export default class Environment {
  static ACTION_DIMS = 0;
  static OBSERVATION_DIMS = 0;

  constructor(name) {
    this.name = name;
  }

  reset() {
  }

  observe() {
    return [];
  }

  actionMask() {
    return [];
  }

  step() {
    return 0;
  }

  isFinished() {
    return false;
  }

  toString() {
    return '';
  }
}
