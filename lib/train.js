import v8 from 'v8';
import path from 'path';

import tf from '@tensorflow/tfjs-node';
import createDebug from 'debug';

import { Avatar } from './model/index.js';

const debug = createDebug('rl:train');

export default class Train {
  constructor(Model, Environment, options) {
    this.Environment = Environment;
    this.options = Object.assign({
      lr: 0.01,
      batchSize: 128,
      maxSteps: 128,
      entropyAlpha: 0.2,
      rewardAlpha: 0.95,
      logDir: path.join('.', 'logs'),

      printEnvEvery: 10,
    }, options);

    this.options.avatar = Object.assign({}, {
      count: 0,
      copyProbability: 0.1,
    }, this.options.avatar);

    const runName = this.options.runName || 'default';

    //
    // Initialize models
    //

    this.model = new Model(Environment);

    this.avatars = [];
    for (let i = 0; i < this.options.avatar.count; i++) {
      this.avatars.push(new Avatar(Environment, Model, {
        source: this.model,
        copyProbability: this.options.avatar.copyProbability,
      }));
    }

    // Build model to initialize weights
    const inputShape = [ this.options.batchSize, Environment.OBSERVATION_DIMS ];
    this.model.build(inputShape);
    for (const model of this.avatars) {
      model.build(inputShape);
    }

    //
    // Initialize environments
    //

    this.envs = [];
    for (let i = 0; i < this.options.batchSize; i++) {
      this.envs.push(new this.Environment(this.options.envOptions));
    }

    this.optimizer = this.options.optimizer ||
      tf.train.rmsprop(this.options.lr);
    this.log = tf.node.summaryFileWriter(
      path.join(this.options.logDir, runName));
  }

  async loop() {
    for (const env of this.envs) {
      env.reset();
    }

    let state = this.model.initialState;

    // Track rewards and history only for `this.model`
    const rewards = new Array(this.envs.length).fill(0);
    const history = [];
    for (let t = 0; t < this.options.maxSteps; t++) {
      state = await this.model.step({
        envs: this.envs,
        state,

        history,
        rewards,
      });

      const allFinished = this.envs.every((env) => env.isFinished());
      if (allFinished) {
        tf.dispose(state);
        break;
      }
    }

    if (state !== this.model.initialState) {
      tf.dispose(state);
    }

    return { finalRewards: rewards, history };
  }

  async runEpoch(epochNum) {
    const { finalRewards, history } = await this.loop();

    //
    // Compute loss & apply gradients
    //

    const train = await this.optimizer.minimize(() => {
      let entropyLoss = tf.tensor(0);
      let policyLoss = tf.tensor(0);
      let valueLoss = tf.tensor(0);

      for (const entry of history) {
        const {
          observation,
          actionMask,
          state = this.model.initialState,
          action,
          rewards,
          value,
        } = entry;

        const { probs, value: valueTensor } = this.model.forward(
          observation, state, actionMask);

        const sampleMask = tf.oneHot(action, this.Environment.ACTION_DIMS);
        const logProbs = probs.mul(sampleMask).sum(-1);

        const entropy = probs.mul(probs.exp()).sum(-1).mul(-1);

        const rewardToGo = finalRewards.slice();
        for (let i = 0; i < rewards.length; i++) {
          rewardToGo[i] -= rewards[i];
        }

        policyLoss = policyLoss.add(
          logProbs.mul(tf.tensor(rewardToGo).sub(tf.tensor(value))).mean(-1));
        valueLoss = valueLoss.add(
          valueTensor.sub(rewardToGo).square().div(2).mean(-1));
        entropyLoss = entropyLoss.add(entropy.mean(-1));
      }
      valueLoss = valueLoss.div(history.length);
      entropyLoss = entropyLoss.div(history.length);
      entropyLoss = entropyLoss.mul(-1);
      policyLoss = policyLoss.mul(-1);

      this.log.scalar('rl/value_loss', valueLoss.dataSync()[0], epochNum);
      this.log.scalar('rl/entropy_loss', entropyLoss.dataSync()[0], epochNum);

      return policyLoss
        .add(valueLoss)
        .add(entropyLoss.mul(this.options.entropyAlpha));
    }, true);

    this.model.evolve();

    const [ loss ] = await train.data();
    tf.dispose(history);
    tf.dispose(train);

    let avg = 0;
    let avgCount = 0;
    for (const reward of finalRewards) {
      avg += reward;
      avgCount++;
    }
    avg /= avgCount;

    const rss = process.memoryUsage().rss / 1024 / 1024;

    this.log.scalar('rl/avg_reward', avg, epochNum);
    this.log.scalar('rl/rss', rss, epochNum);

    debug(`#${epochNum} Loss: ${loss.toFixed(3)} ` +
      `Average: ${avg.toFixed(3)} RSS: ${rss.toFixed(3)}mb`);

    if (epochNum % this.options.printEnvEvery === 0) {
      const randomEnv = this.envs[(Math.random() * this.envs.length) | 0];
      const envStr = randomEnv.toString();
      if (envStr) {
        debug(envStr);
      }
    }

    // Just to help identify the leaks
    if (this.options.dumpHeapEvery &&
        epochNum % this.options.dumpHeapEvery === 0) {
      if (global.gc) {
        global.gc();
      }
      v8.writeHeapSnapshot();
    }
  }

  async run() {
    for (let i = 0; i < Infinity; i++) {
      await this.runEpoch(i);
    }
  }
}
