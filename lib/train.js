import v8 from 'v8';
import path from 'path';

import tf from '@tensorflow/tfjs-node';

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

    const runName = this.options.runName || 'default';

    this.model = new Model(Environment);

    // Build model to initialize weights
    this.model.build([ this.options.batchSize, Environment.OBSERVATION_DIMS ]);

    this.optimizer = this.options.optimizer ||
      tf.train.rmsprop(this.options.lr);
    this.log = tf.node.summaryFileWriter(
      path.join(this.options.logDir, runName));
  }

  async runEpoch(epochNum) {
    const envs = [];
    for (let i = 0; i < this.options.batchSize; i++) {
      envs.push(new this.Environment());
    }

    const { finalRewards, history } = await this.model.loop(envs, {
      maxSteps: this.options.maxSteps,
    });

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

    console.log(`#${epochNum} Loss: ${loss.toFixed(3)} ` +
      `Average: ${avg.toFixed(3)} RSS: ${rss.toFixed(3)}mb`);

    if (epochNum % this.options.printEnvEvery === 0) {
      const envStr = envs[(Math.random() * envs.length) | 0].toString();
      if (envStr) {
        console.log(envStr);
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
