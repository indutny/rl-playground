export default class Environment {
  static ACTION_DIMS = 0;
  static OBSERVATION_DIMS = 0;

  constructor(name) {
    this.name = name;
  }

  reset() {
  }

  async observe() {
    return [];
  }

  actionMask() {
    return [];
  }

  async step() {
    return 0;
  }

  isFinished() {
    return false;
  }

  toString() {
    return '';
  }
}
