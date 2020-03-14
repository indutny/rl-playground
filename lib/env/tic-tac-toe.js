import Environment from './base.js';

const WIDTH = 10;
const HEIGHT = 10;
const LINE = 5;

export default class TicTacToe extends Environment {
  static ACTION_DIMS = WIDTH * HEIGHT + 1;
  static OBSERVATION_DIMS = WIDTH * HEIGHT;
  static PLAYER_COUNT = 2;

  constructor() {
    super('tic-tac-toe');

    this.field = new Array(WIDTH * HEIGHT).fill(0);

    this.lines = [];

    // Vertical lines
    for (let x = 0; x < WIDTH; x++) {
      for (let y = 0; y <= HEIGHT - LINE; y++) {
        this.lines.push({ x, y, dx: 0, dy: 1 });
      }
    }

    // Horizontal lines
    for (let x = 0; x <= WIDTH - LINE; x++) {
      for (let y = 0; y < HEIGHT; y++) {
        this.lines.push({ x, y, dx: 1, dy: 0 });
      }
    }

    // Diagonal lines
    for (let x = 0; x <= WIDTH - LINE; x++) {
      for (let y = 0; y <= HEIGHT - LINE; y++) {
        this.lines.push({ x, y, dx: 1, dy: 1 });
      }
    }
    for (let x = LINE; x < WIDTH; x++) {
      for (let y = 0; y <= HEIGHT - LINE; y++) {
        this.lines.push({ x, y, dx: -1, dy: 1 });
      }
    }

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
    const out = new Array(TicTacToe.OBSERVATION_DIMS);
    for (const [ i, cell ] of this.field.entries()) {
      out[i] = (cell * this.currentPlayer) | 0;
    }
    return out;
  }

  actionMask() {
    const mask = new Array(TicTacToe.ACTION_DIMS).fill(0);

    let cellsLeft = 0;
    if (!this.winner) {
      for (const [ i, cell ] of this.field.entries()) {
        if (cell !== 0) {
          continue;
        }
        cellsLeft++;
        mask[i] = 1;
      }
    }

    if (cellsLeft === 0) {
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
    this.check(this.currentPlayer);
    this.currentPlayer *= -1;

    if (!this.isFinished()) {
      return 0;
    }

    const turn = Math.ceil(this.trace.length / 2);
    const rewardFactor = (turn - LINE + 1) ** 0.2;

    // End of the game
    if (this.winner === -this.currentPlayer) {
      const reward = 1 / rewardFactor;
      this.pendingReward = -reward;
      return reward;
    } else if (this.winner === this.currentPlayer) {
      const reward = -1 / rewardFactor;
      this.pendingReward = -reward;
      return reward;
    }

    // Draw
    return 0;
  }

  check(cell) {
    let winner;

    for (const { x, y, dx, dy } of this.lines) {
      winner = this.checkLine(x, y, dx, dy, cell);
      if (winner) {
        break;
      }
    }
    this.winner = winner;

    if (!this.winner && this.field.every((cell) => cell !== 0)) {
      this.winner = -23;
    }
  }

  checkLine(x, y, dx, dy, cell) {
    for (let i = 0; i < LINE; i++) {
      if (y * WIDTH + x >= this.field.length) {
        throw new Error('Yikes');
      }

      if (this.field[y * WIDTH + x] !== cell) {
        return 0;
      }
      x += dx;
      y += dy;
    }
    return cell;
  }

  isFinished() {
    return this.winner !== 0 && !this.pendingReward;
  }

  toString() {
    const field = new Array(WIDTH * HEIGHT).fill(0);

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

      const out = [];
      for (let y = 0; y < HEIGHT; y++) {
        out.push(str.slice(y * WIDTH, (y + 1) * WIDTH));
      }
      return out.join('\n');
    }

    const out = [];
    for (const step of this.trace) {
      field[step.action] = step.player;
      out.push(fieldToString());
    }
    return out.join('\n--------\n') + `\nwinner=${this.winner}`;
  }
}
