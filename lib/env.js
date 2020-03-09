import maze from '@indutny/maze';

const WIDTH = 15;
const HEIGHT = 15;
const WALK_STEPS = 32;

const EMPTY_CELL = 0;
const WALL_CELL = 1;
const START_CELL = 2;
const FINISH_CELL = -1;
const VISIT_CELL = 3;

function generateField() {
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
  let startX;
  let startY;
  do {
    startX = (Math.random() * WIDTH) | 0;
    startY = (Math.random() * HEIGHT) | 0;
  } while (field[startY][startX] !== EMPTY_CELL);

  const candidates = [];
  function walk(x, y, steps = 0) {
    if (field[y][x] !== EMPTY_CELL) {
      return false;
    }
    field[y][x] = VISIT_CELL;

    const top = walk(x, y - 1, steps + 1);
    const bottom = walk(x, y + 1, steps + 1);
    const left = walk(x - 1, y, steps + 1);
    const right = walk(x + 1, y, steps + 1);

    if (steps >= WALK_STEPS || !top && !bottom && !left && !right) {
      candidates.push({ x, y, steps });
    }

    field[y][x] = EMPTY_CELL;

    return true;
  }
  walk(startX, startY);

  candidates.sort((a, b) => b.steps - a.steps);
  const { x: finishX, y: finishY, steps: finishSteps } = candidates[0];

  field[startY][startX] = START_CELL;
  field[finishY][finishX] = FINISH_CELL;

  return field;
}

function printField(field) {
  for (const row of field) {
    console.log(row.map((cell) => {
      if (cell === EMPTY_CELL) {
        return ' ';
      } else if (cell === WALL_CELL) {
        return '#';
      } else if (cell === FINISH_CELL) {
        return 'F';
      } else if (cell === START_CELL) {
        return 'S';
      } else if (cell === VISIT_CELL) {
        return '*';
      } else {
        return '?';
      }
    }).join(''));
  }
}

export default class Environment {
  static ACTION_DIMS = 4;
  static OBSERVATION_DIMS = 8;

  constructor() {
    this.reset();
  }

  reset() {
    // Copy field
    this.field = generateField();
    this.trace = [];

    this.start = this.find(START_CELL);
    this.finish = this.find(FINISH_CELL);

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

  distance(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
  }

  observe() {
    const { x, y } = this.current;
    const top = this.field[y - 1];
    const middle = this.field[y];
    const bottom = this.field[y + 1];

    return [
      top[x - 1], top[x], top[x + 1],
      middle[x - 1], middle[x + 1],
      bottom[x - 1], bottom[x], bottom[x + 1],
    ];
  }

  step(action) {
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
    this.trace.push(this.current);
    return this.isFinished() ? 100 / this.steps : 0.0;
  }

  isFinished() {
    return this.current.x === this.finish.x &&
      this.current.y === this.finish.y;
  }

  printTrace() {
    for (const { x, y } of this.trace) {
      this.field[y][x] = VISIT_CELL;
    }
    printField(this.field);
    for (const { x, y } of this.trace) {
      this.field[y][x] = EMPTY_CELL;
    }
  }
}
