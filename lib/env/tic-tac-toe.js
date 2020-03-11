import Environment from './base.js';

export default class TicTacToe extends Environment {
  static ACTION_DIMS = 9;
  static OBSERVATION_DIMS = 9;

  constructor() {
    super('tic-tac-toe');

    this.field = new Array(9).fill(0);

    this.reset();
  }

  reset() {
    this.field.fill(0);
    this.currentPlayer = 1;
    this.trace = [];

    this.winner = 0;
  }

  async observe() {
    return this.field;
  }

  actionMask() {
    return this.field.map((cell) => cell === 0 ? 1 : 0);
  }

  async step(action) {
    if (this.isFinished()) {
      return 0;
    }

    this.trace.push({ action: action, player: this.currentPlayer });

    this.field[action] = this.currentPlayer;
    this.currentPlayer *= -1;

    this.check();

    // We won!
    if (this.isFinished() && this.winner === -this.currentPlayer) {
      // Add small discount for winning early
      return 1 - this.trace.length / 50;
    }

    return 0;
  }

  check() {
    if (this.field.every((cell) => cell !== 0)) {
      this.winner = -23;
      return;
    }

    this.winner = this.checkLine(0, 0, 1, 0) ||
      this.checkLine(0, 1, 1, 0) ||
      this.checkLine(0, 2, 1, 0) ||
      this.checkLine(0, 0, 0, 1) ||
      this.checkLine(1, 0, 0, 1) ||
      this.checkLine(2, 0, 0, 1) ||
      this.checkLine(0, 0, 1, 1) ||
      this.checkLine(2, 0, -1, 1);
  }

  checkLine(x, y, dx, dy) {
    let cell = this.field[y * 3 + x];
    if (cell === 0) {
      return 0;
    }

    for (let i = 1; i < 3; i++) {
      x += dx;
      y += dy;
      if (this.field[y * 3 + x] !== cell) {
        return 0;
      }
    }
    return cell;
  }

  isFinished() {
    return this.winner !== 0;
  }

  toString() {
    const field = new Array(9).fill(0);

    function fieldToString() {
      const str = field.map((cell) => {
        if (cell === 0) {
          return ' ';
        } else if (cell === 1) {
          return 'X';
        } else if (cell === -1) {
          return 'O';
        }
      }).join('');

      return [
        str.slice(0, 3),
        str.slice(3, 6),
        str.slice(6),
      ].join('\n');
    }

    const out = [];
    for (const step of this.trace) {
      field[step.action] = step.player;
      out.push(fieldToString());
    }
    return out.join('\n--------\n');
  }
}
