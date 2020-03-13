import Environment from './base.js';

export default class TicTacToe extends Environment {
  static ACTION_DIMS = 10;
  static OBSERVATION_DIMS = 10;
  static PLAYER_COUNT = 2;

  constructor() {
    super('tic-tac-toe');

    this.field = new Array(9).fill(0);

    this.reset();
  }

  reset() {
    this.field.fill(0);
    this.currentPlayer = 1;
    this.pendingReward = 0;
    this.trace = [];

    this.winner = 0;
  }

  async observe() {
    const out = new Array(TicTacToe.OBSERVATION_DIMS).fill(0);
    for (const [ i, cell ] of this.field.entries()) {
      if (cell !== 0) {
        out[i] = cell * this.currentPlayer;
      }
    }
    out[out.length - 1] = this.currentPlayer;
    return out;
  }

  actionMask() {
    const mask = new Array(TicTacToe.ACTION_DIMS).fill(0);
    let some = false;
    if (!this.winner) {
      for (const [ i, cell ] of this.field.entries()) {
        if (cell !== 0) {
          continue;
        }
        some = true;
        mask[i] = 1;
      }
    }

    if (!some) {
      mask[mask.length - 1] = 1;
    }
    return mask;
  }

  async step(action) {
    if (this.isFinished()) {
      return 0;
    }

    if (this.pendingReward) {
      const reward = this.pendingReward;
      this.pendingReward = 0;
      return reward;
    }

    if (action >= this.field.length) {
      return 0;
    }

    this.trace.push({ action: action, player: this.currentPlayer });

    this.field[action] = this.currentPlayer;
    this.currentPlayer *= -1;

    this.check();

    if (!this.isFinished()) {
      return 0;
    }

    // End of the game
    if (this.winner === -this.currentPlayer) {
      this.pendingReward = -1;
      return 1;
    } else if (this.winner === this.currentPlayer) {
      this.pendingReward = 1;
      return -1;
    }

    // Draw
    return 0;
  }

  check() {
    this.winner = this.checkLine(0, 0, 1, 0) ||
      this.checkLine(0, 1, 1, 0) ||
      this.checkLine(0, 2, 1, 0) ||
      this.checkLine(0, 0, 0, 1) ||
      this.checkLine(1, 0, 0, 1) ||
      this.checkLine(2, 0, 0, 1) ||
      this.checkLine(0, 0, 1, 1) ||
      this.checkLine(2, 0, -1, 1);

    if (!this.winner && this.field.every((cell) => cell !== 0)) {
      this.winner = -23;
    }
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
    return this.winner !== 0 && !this.pendingReward;
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
