import v8 from 'v8';
import path from 'path';

import tf from '@tensorflow/tfjs-node';
import createDebug from 'debug';

import { Avatar } from './model/index.js';
import { shuffle } from './utils.js';

const debug = createDebug('rl:train');

const EPSILON = 1e-23;

export default class Train {
  constructor(Model, Environment, options) {
    this.Environment = Environment;
    this.options = Object.assign({
      lr: 0.01,
      batchSize: 128,
      maxSteps: 128,
      entropyAlpha: 0.2,
      rewardGamma: 0.99,
      logDir: path.join('.', 'logs'),

      printEnvEvery: 10,

      ppo: { epsilon: 0.2 },
    }, options);

    this.options.avatar = Object.assign({}, {
      // Total number of avatars for whole training
      totalCount: 0,

      // Number of avatars playing during each game loop
      count: 0,

      copyProbability: 0.05,
    }, this.options.avatar);

    const runName = this.options.runName || 'default';

    //
    // Initialize models
    //

    this.model = new Model(Environment);

    this.avatars = [];
    for (let i = 0; i < this.options.avatar.totalCount; i++) {
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

    const loopAvatars = this.avatars.slice();
    shuffle(loopAvatars);

    const models = [ this.model ].concat(
      loopAvatars.slice(0, this.options.avatar.count));

    // Shuffle models to guarantee random order!
    // NOTE: The downside is that the order is fixed for the whole batch
    shuffle(models);

    const initialStates = models.map((model) => model.initialState);
    let states = initialStates;

    // Track history only for `this.model`
    const history = [];
    for (let t = 0; t < this.options.maxSteps; t++) {
      const newStates = [];
      for (const [ i, model ] of models.entries()) {
        const isMain = model === this.model;

        const newState = await model.step({
          envs: this.envs,
          state: states[i],

          history: isMain ? history : null,
        });
        newStates.push(newState);
      }
      states = newStates;

      const allFinished = this.envs.every((env) => env.isFinished());
      if (allFinished) {
        tf.dispose(states);
        break;
      }
    }

    if (states !== initialStates) {
      tf.dispose(states);
    }

    return history;
  }

  computeLoss(history, metrics) {
    let entropyGain = tf.tensor(0);
    let policyGain = tf.tensor(0);
    let valueLoss = tf.tensor(0);

    // 2d array: [ maxSteps, batchSize ]
    const rewardsToGo = history.map(({ rewards }) => rewards.slice());

    for (let i = rewardsToGo.length - 2; i >= 0; i--) {
      const factor = Math.pow(this.options.rewardGamma,
        rewardsToGo.length - 1 - i);
      const stepRewards = rewardsToGo[i];

      for (let j = 0; j < stepRewards.length; j++) {
        // Discount past
        stepRewards[j] *= factor;

        // Add future to the present
        stepRewards[j] += rewardsToGo[i + 1][j];
      }
    }

    let state = this.model.initialState;
    for (const [ step, entry ] of history.entries()) {
      const {
        observation,
        actionMask,
        logProbs,
        action,
        value,
      } = entry;

      let {
        logProbs: liveLogProbs,
        value: liveValue,
        newState,
      } = this.model.forward(observation, state, actionMask);
      state = newState;

      // Compute entropy
      const entropy = liveLogProbs.mul(liveLogProbs.exp()).sum(-1).mul(-1);
      entropyGain = entropyGain.add(entropy.mean(-1));

      // Mask out actions that we did not take
      const sampleMask = tf.oneHot(action, this.Environment.ACTION_DIMS);
      const actionLogProb = {
        past: logProbs.mul(sampleMask).sum(-1),
        live: liveLogProbs.mul(sampleMask).sum(-1),
      };

      // Advantage is a difference between future and present rewards and
      // predicted rewards
      const rewards = tf.tensor(rewardsToGo[step]);
      const advantage = rewards.sub(tf.tensor(value));

      // Predict rewards better!
      valueLoss = valueLoss.add(
        liveValue.sub(rewards).square().div(2).mean(-1));

      // Finally, optimize the policy itself
      if (this.options.ppo) {
        const actionProb = {
          past: actionLogProb.past.exp(),
          live: actionLogProb.live.exp(),
        };

        const lhs = actionProb.live.div(actionProb.past.add(EPSILON));
        const rhs = advantage.sign().mul(this.options.ppo.epsilon).add(1);

        const min = tf.minimum(lhs.mul(advantage), rhs.mul(advantage));
        policyGain = policyGain.add(min.mean(-1));
      } else {
        // Vanilla RL
        policyGain = policyGain.add(actionLogProb.live.mul(advantage).mean(-1));
      }
    }
    valueLoss = valueLoss.div(history.length);
    entropyGain = entropyGain.div(history.length);
    policyGain = policyGain.div(history.length);

    // We want to maximize these:
    const entropyLoss = entropyGain.mul(-1);
    const policyLoss = policyGain.mul(-1);

    const totalRewards = rewardsToGo[0];

    let averageReward = 0;
    for (const reward of totalRewards) {
      averageReward += reward;
    }
    averageReward /= totalRewards.length;

    const loss = policyLoss
      .add(valueLoss)
      .add(entropyLoss.mul(this.options.entropyAlpha));

    metrics.set('value_loss', valueLoss.dataSync()[0]);
    metrics.set('entropy_loss', entropyLoss.dataSync()[0]);
    metrics.set('reward', averageReward);

    return loss;
  }

  async runEpoch(epochNum) {
    const history = await this.loop();

    //
    // Compute loss & apply gradients
    //

    const metrics = new Map();

    const train = await this.optimizer.minimize(() => {
      return this.computeLoss(history, metrics);
    }, true);

    const [ loss ] = await train.data();
    tf.dispose(history);
    tf.dispose(train);

    // Increase generation count
    this.model.evolve();

    const rss = process.memoryUsage().rss / 1024 / 1024;
    metrics.set('rss', rss);

    //
    // Report metrics and log debug information
    //

    for (const [ key, value ] of metrics) {
      this.log.scalar(`rl/${key}`, value, epochNum);
    }

    debug(`#${epochNum} Loss: ${loss.toFixed(3)} ` +
      `Reward: ${metrics.get('reward').toFixed(3)} RSS: ${rss.toFixed(3)}mb`);

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
