#!/usr/bin/env node
import yargs from 'yargs';
import path from 'path';

import { RNNModel } from '../lib/model/index.js';
import {
  Maze, Sequence, Bandit, TicTacToe, Prisoner,
} from '../lib/env/index.js';
import Train from '../lib/train.js';

const argv = yargs
  .option('lr', { type: 'number', default: 0.01 })
  .option('train-multiplier', { type: 'number', default: 16 })
  .option('batch-size', { type: 'number', default: 128 })
  .option('max-steps', { type: 'number', default: 128 })
  .option('entropy-alpha', { type: 'number', default: 0.2 })
  .option('entropy-decay', { type: 'number', default: 2000 })
  .option('reward-gamma', { type: 'number', default: 0.99 })
  .option('log-dir', { type: 'string', default: path.join('.', 'logs') })
  .option('save-dir', { type: 'string', default: path.join('.', 'saves') })
  .option('name', { type: 'string', default: `default-${Date.now()}` })
  .option('print-env-every', { type: 'number', default: 10 })
  .option('dump-heap-every', { type: 'number', default: 0 })
  .option('save-every', { type: 'number', default: 100 })
  .option('max-saves', { type: 'number', default: 100 })
  .argv;

const t = new Train(RNNModel, Maze, {
  lr: argv.lr,
  trainMultiplier: argv['train-multiplier'],

  batchSize: argv['batch-size'],
  maxSteps: argv['max-steps'],
  entropyAlpha: argv['entropy-alpha'],
  entropyDecay: argv['entropy-decay'],
  rewardGamma: argv['reward-gamma'],
  logDir: argv['log-dir'],
  runName: argv['name'],

  printEnvEvery: argv['print-env-every'],
  dumpHeapEvery: argv['dump-heap-every'],
  saveEvery: argv['save-every'],
  saveDir: argv['save-dir'],
  maxSaves: argv['max-saves'],

  modelCount: 1,

  avatar: {
    count: 0,
    generationSpread: 10,
  }
});

t.run().catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
