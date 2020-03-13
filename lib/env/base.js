export default class Environment {
  static ACTION_DIMS = 0;
  static OBSERVATION_DIMS = 0;
  static PLAYER_COUNT = 1;

  constructor(name) {
    this.name = name;
  }

  reset() {
  }

  getBaton() {
    return undefined;
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
