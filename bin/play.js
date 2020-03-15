#!/usr/bin/env node
import yargs from 'yargs';
import { promises as fs } from 'fs';

import { RNNModel } from '../lib/model/index.js';
import {
  Maze, Sequence, Bandit, TicTacToe, Prisoner,
} from '../lib/env/index.js';

const argv = yargs
  .option('save', { type: 'string', required: true })
  .option('max-steps', { type: 'number', default: 32 })
  .argv;

async function main(Model, Environment) {
  const env = new Environment();
  const model = new Model(Environment);

  model.build([ Environment.OBSERVATION_DIMS ]);

  const raw = await fs.readFile(argv.save);
  await model.loadJSON(raw.toString());

  let state = model.initialState;
  for (let i = 0; i < argv['max-steps']; i++) {
    state = await model.step({
      envs: [ env ],
      state,
      deterministic: false,
    });

    if (env.isFinished()) {
      break;
    }
  }

  console.log(env.toString());
}

main(RNNModel, Maze).catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
