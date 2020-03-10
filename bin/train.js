#!/usr/bin/env node
import yargs from 'yargs';
import path from 'path';

import { RNNModel } from '../lib/model/index.js';
import { Maze, Sequence, Bandit } from '../lib/env/index.js';
import Train from '../lib/train.js';

const argv = yargs
  .option('lr', { type: 'number', default: 0.01 })
  .option('batch-size', { type: 'number', default: 128 })
  .option('max-steps', { type: 'number', default: 128 })
  .option('entropy-alpha', { type: 'number', default: 0.2 })
  .option('log-dir', { type: 'string', default: path.join('.', 'logs') })
  .option('name', { type: 'string', default: 'default' })
  .option('print-env-every', { type: 'number', default: 100 })
  .option('dump-heap-every', { type: 'number', default: 0 })
  .argv;

const t = new Train(RNNModel, Sequence, {
  lr: argv.lr,
  batchSize: argv['batch-size'],
  maxSteps: argv['max-steps'],
  entropyAlpha: argv['entropy-alpha'],
  logDir: argv['log-dir'],
  runName: argv['name'],

  printEnvEvery: argv['print-env-every'],
  dumpHeapEvery: argv['dump-heap-every'],
});

t.run().catch((e) => {
  console.error(e.stack);
  process.exit(1);
});