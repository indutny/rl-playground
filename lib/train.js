import path from 'path';
import tf from '@tensorflow/tfjs-node';

import Model from './model.js';
import { Maze, Sequence, Bandit } from './env/index.js';

class Train {
  constructor(Environment, options) {
    this.Environment = Environment;
    this.options = Object.assign({
      lr: 0.01,
      batchSize: 128,
      maxSteps: 128,
      entropyAlpha: 0.2,
      logDir: path.join('.', 'logs'),

      printEnvEvery: 10,
    }, options);

    const runName = this.options.runName || 'default';

    this.model = new Model(Environment);

    // Build model to initialize weights
    this.model.build([ this.options.batchSize, Environment.OBSERVATION_DIMS ]);

    this.optimizer = tf.train.rmsprop(this.options.lr);
    this.log = tf.node.summaryFileWriter(
      path.join(this.options.logDir, runName));
  }

  async runEpoch(epochNum) {
    const envs = [];
    for (let i = 0; i < this.options.batchSize; i++) {
      envs.push(new this.Environment());
    }

    // Tensors to cleanup
    const cleanup = [];

    //
    // Interact with environment
    //

    const history = [];
    const rewards = new Array(envs.length).fill(0);
    let state = this.model.initialState;
    for (let t = 0; t < this.options.maxSteps; t++) {
      const observation = envs.map((env) => env.observe());
      const actionMask = envs.map((env) => env.actionMask());
      const {
        action: actionTensor,
        newState,
        value: valueTensor,
      } = tf.tidy(() => {
        return this.model.sample(
          tf.tensor(observation), state, tf.tensor(actionMask));
      });

      const [ action, value ] = await Promise.all([
        actionTensor.data(),
        valueTensor.data(),
      ]);
      cleanup.push(newState, valueTensor, actionTensor);

      history.push({
        observation,
        actionMask,
        state,
        action,
        rewards: rewards.slice(),
        value,
      });

      // Update environment
      let allFinished = true;
      for (const [ i, env ] of envs.entries()) {
        rewards[i] += env.step(action[i]);
        if (!env.isFinished()) {
          allFinished = false;
        }
      }

      if (allFinished) {
        break;
      }

      state = newState;
    }

    const finalRewards = rewards;

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
          state,
          action,
          rewards,
          value,
        } = entry;

        const { probs, value: valueTensor } = this.model.forward(
          tf.tensor(observation),
          state, tf.tensor(actionMask));

        const sampleMask = tf.oneHot(action, this.Environment.ACTION_DIMS);
        const entropy = probs.mul(probs.exp()).sum(-1).mul(-1);
        const logProbs = probs.mul(sampleMask).sum(-1);

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
    tf.dispose(cleanup);
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
      console.log(envStr);
    }
  }

  async run() {
    for (let i = 0; i < Infinity; i++) {
      await this.runEpoch(i);
    }
  }
}

const t = new Train(Sequence, {
  runName: process.env.RUN_NAME,
});

t.run().catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
