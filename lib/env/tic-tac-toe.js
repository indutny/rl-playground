import Environment from './base.js';

export default class TicTacToe extends Environment {
  static ACTION_DIMS = 9;
  static OBSERVATION_DIMS = 9;

  #isFinished = false;

  constructor() {
    super('bandit');

    this.field = new Array(9).fill(0);
    this.players = new Map();

    this.reset(0);
  }

  reset(seed) {
    this.field.fill(0);
    this.players.clear();
    this.#isFinished = false;
    this.currentPlayer = seed > 0.5 ? 1 : -1;

    this.resolveOpponent = null;
  }

  getBaton(model) {
    if (this.players.size > 2) {
      throw new Error('Way too many players!');
    }

    this.players.set(model, this.players.size === 0 ? -1 : 1);
    return model;
  }

  async observe(baton) {
    if (!this.players.has(baton)) {
      throw new Error('Invalid baton!');
    }
    const player = this.players.get(baton);

    if (player !== this.currentPlayer) {
      await new Promise((resolve) => {
        this.resolveOpponent = resolve;
      });
    }

    return this.field;
  }

  actionMask() {
    return this.field.map((cell) => cell === 0 ? 1 : 0);
  }

  async step(action, baton) {
    if (!this.players.has(baton)) {
      throw new Error('Invalid baton!');
    }
    const player = this.players.get(baton);

    if (player !== this.currentPlayer) {
      throw new Error('Unexpected step from a wrong player!');
    }

    if (this.resolveOpponent) {
      this.resolveOpponent();
      this.resolveOpponent = null;
    }

    if (this.isFinished()) {
      return 0;
    }

    this.currentPlayer *= -1;

    return 0;
  }

  isFinished() {
    return this.#isFinished;
  }
}
