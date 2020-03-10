import maze from '@indutny/maze';

import Environment from './base.js';

const WIDTH = 31;
const HEIGHT = 31;

const FOOD_COUNT = 32;

const EMPTY_CELL = 0;
const WALL_CELL = 1;
const START_CELL = 2;
const FOOD_CELL = -1;
const VISIT_CELL = 3;

function generateField(maxSteps) {
  const field = maze({
    width: WIDTH - 2,
    height: HEIGHT - 2,
    empty: EMPTY_CELL,
    wall: WALL_CELL,
  });

  // Add walls
  for (const row of field) {
    row.unshift(WALL_CELL);
    row.push(WALL_CELL);
  }
  field.unshift(new Array(WIDTH).fill(WALL_CELL));
  field.push(new Array(WIDTH).fill(WALL_CELL));

  // Pick starting point
  function findEmpty() {
    let x;
    let y;
    do {
      x = (Math.random() * WIDTH) | 0;
      y = (Math.random() * HEIGHT) | 0;
    } while (field[y][x] !== EMPTY_CELL);
    return { x, y };
  }

  const { x: startX, y: startY } = findEmpty();
  field[startY][startX] = START_CELL;

  // Scatter some food
  for (let i = 0; i < FOOD_COUNT; i++) {
    const { x: foodX, y: foodY } = findEmpty();
    field[foodY][foodX] = FOOD_CELL;
  }

  return field;
}

function stringifyField(field) {
  const out = [];
  for (const row of field) {
    const line = row.map((cell) => {
      if (cell === EMPTY_CELL) {
        return ' ';
      } else if (cell === WALL_CELL) {
        return '#';
      } else if (cell === FOOD_CELL) {
        return 'F';
      } else if (cell === START_CELL) {
        return 'S';
      } else if (cell === VISIT_CELL) {
        return '*';
      } else {
        return '?';
      }
    });

    out.push(line.join(''));
  }
  return out.join('\n');
}

const DEFAULT_MAX_STEPS = 64;

export default class Maze extends Environment {
  static ACTION_DIMS = 4;
  static OBSERVATION_DIMS = 8;

  constructor() {
    super('maze');

    this.reset();
  }

  reset() {
    this.field = generateField();
    // Copy field
    this.visited = this.field.map((row) => row.slice().fill(false));
    this.trace = [];

    this.start = this.find(START_CELL);

    // Clear start marker
    this.field[this.start.y][this.start.x] = 0;

    this.current = { x: this.start.x, y: this.start.y };
    this.steps = 0;
  }

  find(cell) {
    for (let y = 0; y < this.field.length; y++) {
      const row = this.field[y];
      for (let x = 0; x < row.length; x++) {
        if (row[x] === cell) {
          return { x, y };
        }
      }
    }
    throw new Error('Cell not found!');
  }

  ray(x, y, dx, dy, cell) {
    let steps = 0;
    while (this.field[y][x] !== cell) {
      if (this.field[y][x] === WALL_CELL) {
        return -1;
      }
      y += dy;
      x += dx;
      steps++;
    }
    return steps;
  }

  isEmpty(x, y) {
    return this.field[y][x] === EMPTY_CELL;
  }

  isNonWall(x, y) {
    return this.field[y][x] !== WALL_CELL;
  }

  isVisited(x, y) {
    return this.visited[y][x];
  }

  async observe() {
    const { x, y } = this.current;

    return [
      // Walls
      this.ray(x, y, -1, 0, WALL_CELL) / WIDTH,
      this.ray(x, y, 1, 0, WALL_CELL) / WIDTH,
      this.ray(x, y, 0, -1, WALL_CELL) / HEIGHT,
      this.ray(x, y, 0, 1, WALL_CELL) / HEIGHT,

      // Food
      this.ray(x, y, -1, 0, FOOD_CELL) / WIDTH,
      this.ray(x, y, 1, 0, FOOD_CELL) / WIDTH,
      this.ray(x, y, 0, -1, FOOD_CELL) / HEIGHT,
      this.ray(x, y, 0, 1, FOOD_CELL) / HEIGHT,
    ];
  }

  actionMask() {
    const { x, y } = this.current;
    return [
      this.isNonWall(x, y - 1) ? 1 : 0,
      this.isNonWall(x, y + 1) ? 1 : 0,
      this.isNonWall(x - 1, y) ? 1 : 0,
      this.isNonWall(x + 1, y) ? 1 : 0,
    ];
  }

  async step(action) {
    if (this.isFinished()) {
      return 0;
    }
    this.steps++;

    let { x, y } = this.current;
    if (action === 0) {
      y -= 1;
    } else if (action === 1) {
      y += 1;
    } else if (action === 2) {
      x -= 1;
    } else if (action === 3) {
      x += 1;
    }

    if (this.field[y][x] === WALL_CELL) {
      return 0;
    }

    const visited = this.visited[y][x];
    this.visited[y][x] = true;

    this.current = { x, y };
    this.trace.push(this.current);

    if (this.field[y][x] === FOOD_CELL) {
      this.field[y][x] = EMPTY_CELL;
      return 1;
    }

    // Slightly encourage exploration
    return visited ? 0 : 0.05;
  }

  isFinished() {
    return false;
  }

  toString() {
    for (const { x, y } of this.trace) {
      this.field[y][x] = VISIT_CELL;
    }
    const out = stringifyField(this.field);
    for (const { x, y } of this.trace) {
      this.field[y][x] = EMPTY_CELL;
    }
    return out;
  }
}
