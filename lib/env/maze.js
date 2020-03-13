import maze from '@indutny/maze';

import Environment from './base.js';

const WIDTH = 31;
const HEIGHT = 31;

const FOOD_EVERY = 6;
const PASSAGES = Math.floor(WIDTH * HEIGHT / 16);

const EMPTY_CELL = 0;
const WALL_CELL = 1;
const START_CELL = 2;
const FOOD_CELL = -1;
const VISIT_CELL = 3;
const EATEN_CELL = 4;

function computeDistances(field, x, y, distanceField, step = 0) {
  if (field[y][x] !== EMPTY_CELL) {
    return;
  }

  const old = distanceField[y][x];
  if (old <= step) {
    return;
  }

  distanceField[y][x] = step;
  computeDistances(field, x - 1, y, distanceField, step + 1);
  computeDistances(field, x + 1, y, distanceField, step + 1);
  computeDistances(field, x, y - 1, distanceField, step + 1);
  computeDistances(field, x, y + 1, distanceField, step + 1);
}

function getLongestPath(distanceField) {
  let max = 0;
  let maxX = null;
  let maxY = null;
  for (let y = 0; y < distanceField.length; y++) {
    const row = distanceField[y];
    for (let x = 0; x < row.length; x++) {
      const cell = row[x];

      if (isFinite(cell) && cell > max) {
        max = cell;
        maxX = x;
        maxY = y;
      }
    }
  }

  let distance = 0;
  let path = [];
  while (max > 0) {
    path.push({ x: maxX, y: maxY });
    distance++;

    max--;
    if (distanceField[maxY][maxX - 1] === max) {
      maxX--;
    } else if (distanceField[maxY][maxX + 1] === max) {
      maxX++;
    } else if (distanceField[maxY - 1][maxX] === max) {
      maxY--;
    } else if (distanceField[maxY + 1][maxX] === max) {
      maxY++;
    }
  }

  return { path, distance };
}

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
  function find(cell) {
    const cells = [];
    for (let y = 1; y < HEIGHT - 1; y++) {
      const row = field[y];
      for (let x = 1; x < WIDTH - 1; x++) {
        if (row[x] === cell) {
          cells.push({ x, y });
        }
      }
    }
    return cells[(cells.length * Math.random()) | 0];
  }

  // Don't let player become trapped
  const { x: startX, y: startY } = find(EMPTY_CELL);

  // Build extra passages
  for (let i = 0; i < PASSAGES; i++) {
    const x = (Math.random() * (WIDTH - 2)) | 0 + 1;
    const y = (Math.random() * (HEIGHT - 2)) | 0 + 1;
    field[y][x] = EMPTY_CELL;
  }

  const distanceField = [];
  for (const row of field) {
    distanceField.push(row.slice().fill(Infinity));
  }

  computeDistances(field, startX, startY, distanceField);
  const { path } = getLongestPath(distanceField);

  for (let y = 0; y < distanceField.length; y++) {
    const distanceRow = distanceField[y];
    const row = field[y];
    for (let x = 0; x < row.length; x++) {
      const distance = distanceRow[x];

      if (isFinite(distance) && distance % FOOD_EVERY === 0) {
        field[y][x] = FOOD_CELL;
      }
    }
  }

  field[startY][startX] = START_CELL;

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
      } else if (cell === EATEN_CELL) {
        return 'X';
      } else {
        return '?';
      }
    });

    out.push(line.join(''));
  }
  return out.join('\n');
}

const DEFAULT_MAX_STEPS = 64;
const OBSERVATION_RADIUS = 3;

export default class Maze extends Environment {
  static ACTION_DIMS = 4;
  static OBSERVATION_DIMS = (2 * OBSERVATION_RADIUS + 1) **2 ;
  static PLAYER_COUNT = 1;

  constructor() {
    super('maze');

    this.reset();
  }

  reset() {
    this.field = generateField();
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

  isNonWall(x, y) {
    return this.field[y][x] !== WALL_CELL;
  }

  cell(x, y) {
    if (y < 0 || y >= HEIGHT || x < 0 || x >= WIDTH) {
      return WALL_CELL;
    }
    return this.field[y][x];
  }

  async observe() {
    const { x, y } = this.current;

    const out = new Array(Maze.OBSERVATION_DIMS).fill(0);
    for (let dx = -OBSERVATION_RADIUS; dx <= OBSERVATION_RADIUS; dx++) {
      for (let dy = -OBSERVATION_RADIUS; dy <= OBSERVATION_RADIUS; dy++) {
        const off = (OBSERVATION_RADIUS + dy) * (2 * OBSERVATION_RADIUS + 1) +
          OBSERVATION_RADIUS + dx;
        out[off] = this.cell(x + dx, y + dy);
      }
    }
    return out;
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

    this.current = { x, y };

    const isFood = this.field[y][x] === FOOD_CELL;
    this.trace.push({ position: this.current, isFood });

    const visited = this.visited[y][x];
    this.visited[y][x] = true;

    if (isFood) {
      this.field[y][x] = EMPTY_CELL;
      return 1;
    }

    // Discourage revisiting cells
    return visited ? -0.1 : 0;
  }

  isFinished() {
    return false;
  }

  toString() {
    for (const { position: { x, y }, isFood } of this.trace) {
      if (isFood) {
        this.field[y][x] = EATEN_CELL;
      } else {
        this.field[y][x] = VISIT_CELL;
      }
    }
    this.field[this.start.y][this.start.x] = START_CELL;

    const out = stringifyField(this.field);
    for (const { position: { x, y } } of this.trace) {
      this.field[y][x] = EMPTY_CELL;
    }
    this.field[this.start.y][this.start.x] = EMPTY_CELL;
    return out;
  }
}
