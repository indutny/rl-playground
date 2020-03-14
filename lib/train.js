import v8 from 'v8';
import path from 'path';
import { promises as fs } from 'fs';

import tf from '@tensorflow/tfjs-node';
import createDebug from 'debug';

import { Avatar } from './model/index.js';
import { shuffle } from './utils.js';

const debug = createDebug('rl:train');

const EPSILON = 1e-23;
const PADDED_EPOCH_LEN = 8;

export default class Train {
  constructor(Model, Environment, options) {
    this.Environment = Environment;
    this.options = Object.assign({
      lr: 0.01,
      trainMultiplier: 16,

      batchSize: 128,
      maxSteps: 128,
      entropyAlpha: 0.2,
      entropyDecay: 2000,
      rewardGamma: 0.95,
      logDir: path.join('.', 'logs'),
      saveDir: path.join('.', 'saves'),

      modelCount: 1,

      printEnvEvery: 10,
      saveEvery: 100,
      maxSaves: 100,

      ppo: { epsilon: 0.2 },
    }, options);

    this.options.avatar = Object.assign({}, {
      // Number of avatars playing during each game loop
      count: 0,

      generationSpread: 100,
    }, this.options.avatar);

    const runName = this.options.runName || 'default';

    //
    // Initialize models
    //

    this.models = [];
    for (let i = 0; i < this.options.modelCount; i++) {
      this.models.push(new Model(Environment));
    }

    this.avatars = [];
    {
      const { generationSpread, count } = this.options.avatar;
      for (let i = 0; i < count; i++) {
        this.avatars.push(new Avatar(Environment, Model, {
          source: this.models[i % this.models.length],
          generationOff: i * generationSpread,
          generationMod: count * generationSpread,
        }));
      }
    }

    // Build model to initialize weights
    const inputShape = [ Environment.OBSERVATION_DIMS ];
    for (const model of this.models.concat(this.avatars)) {
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

    const trainableModels = this.models.slice();
    shuffle(trainableModels);

    const mainModel = trainableModels.pop();

    const opponents = this.avatars.slice().concat(trainableModels);
    shuffle(opponents);

    const models = [ mainModel ].concat(
      opponents.slice(0, this.Environment.PLAYER_COUNT - 1));
    if (models.length !== this.Environment.PLAYER_COUNT) {
      throw new Error('Not enough models! Add more models or avatars');
    }

    // Shuffle models to guarantee random order!
    // NOTE: The downside is that the order is fixed for the whole batch
    shuffle(models);

    const initialStates = models.map((model) => model.initialState);
    let states = initialStates;

    // Track history only for `mainModel`
    const history = [];
    for (let t = 0; t < this.options.maxSteps; t++) {
      const newStates = [];
      for (const [ i, model ] of models.entries()) {
        const isMain = model === mainModel;

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

    return { history, model: mainModel };
  }

  computeRewards(history) {
    // 2d array: [ maxSteps, batchSize ]
    const rewardsToGo = history.map(({ rewards }) => rewards.slice());

    for (let i = rewardsToGo.length - 2; i >= 0; i--) {
      const presentRewards = rewardsToGo[i];
      const futureRewards = rewardsToGo[i + 1];

      for (let j = 0; j < presentRewards.length; j++) {
        presentRewards[j] += this.options.rewardGamma * futureRewards[j];
      }
    }

    return rewardsToGo.map((rewards) => tf.tensor(rewards));
  }

  computeLoss(epochNum, model, rewardsToGo, history, metrics) {
    let entropyGain = tf.tensor(0);
    let policyGain = tf.tensor(0);
    let valueLoss = tf.tensor(0);

    const entropyFactor = 0.5 ** (epochNum / this.options.entropyDecay);

    let state = model.initialState;
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
      } = model.forward(observation, state, actionMask);
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
      const rewards = rewardsToGo[step];
      let advantage = rewards.sub(tf.tensor(value));

      // Normalize advatanges
      {
        const mean = advantage.mean(-1);
        const stddev = advantage.square().mean(-1).sub(mean.square()).sqrt();

        advantage = advantage.sub(mean).div(stddev.add(EPSILON));
      }

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
    entropyGain = entropyGain.div(history.length).mul(entropyFactor);
    policyGain = policyGain.div(history.length);

    // We want to maximize these:
    const entropyLoss = entropyGain.mul(-1);
    const policyLoss = policyGain.mul(-1);

    // Compute undiscounted total rewards for logging
    const totalRewards = history[0].rewards.slice();
    for (const { rewards } of history) {
      for (let i = 0; i < totalRewards.length; i++) {
        totalRewards[i] += rewards[i];
      }
    }

    let averageReward = 0;
    for (const reward of totalRewards) {
      averageReward += reward;
    }
    averageReward /= totalRewards.length;

    const loss = policyLoss
      .add(valueLoss)
      .add(entropyLoss.mul(this.options.entropyAlpha));

    metrics.set('value_loss', valueLoss.dataSync()[0]);
    metrics.set('policy_loss', policyLoss.dataSync()[0]);
    metrics.set('entropy_loss', entropyLoss.dataSync()[0]);
    metrics.set('entropy_factor', entropyFactor);
    metrics.set('reward', averageReward);

    return loss;
  }

  async save(epochNum) {
    const json = await this.models[0].toJSON(epochNum);

    const saveSubDir = path.join(this.options.saveDir, this.options.runName);
    const files = await fs.readdir(saveSubDir);
    const saves = files.filter((file) => /\.save\.json$/.test(file));

    saves.sort();
    if (saves.length === this.options.maxSaves) {
      const oldSave = path.join(saveSubDir, saves[0]);
      debug(`Deleting old save: ${oldSave}`);
      try {
        await fs.unlink(oldSave);
      } catch (e) {
        debug(`Failed to remove old save: ${oldSave}, error: ${e.stack}`);
      }
    }

    let paddedEpoch = epochNum.toString();
    while (paddedEpoch.length < PADDED_EPOCH_LEN) {
      paddedEpoch = '0' + paddedEpoch;
    }

    const filename = path.join(saveSubDir, `${paddedEpoch}.save.json`);

    await fs.writeFile(filename, json);
    debug(`Saved model to ${filename}`);
  }

  async runEpoch(epochNum) {
    debug(`#${epochNum} Loop Start`);

    const { history, model } = await this.loop();
    const rewards = this.computeRewards(history);

    debug(`#${epochNum} Loop End`);

    //
    // Compute loss & apply gradients
    //

    const meanMetrics = new Map();

    for (let i = 0; i < this.options.trainMultiplier; i++) {
      const metrics = new Map();

      const train = await this.optimizer.minimize(() => {
        return this.computeLoss(epochNum, model, rewards, history, metrics);
      }, true);

      const [ loss ] = await train.data();
      tf.dispose(train);

      for (const [ key, value ] of metrics) {
        meanMetrics.set(key,
          (meanMetrics.get(key) || 0) + value / this.options.trainMultiplier);
      }
    }
    tf.dispose(history);

    // Increase generation count
    model.evolve();

    // Update avatars
    for (const avatar of this.avatars) {
      avatar.evolve();
    }

    //
    // Report metrics and log debug information
    //

    const rss = process.memoryUsage().rss / 1024 / 1024;
    meanMetrics.set('rss', rss);

    for (const [ key, value ] of meanMetrics) {
      this.log.scalar(`rl/${key}`, value, epochNum);
    }

    debug(`#${epochNum} Loss: ${meanMetrics.get('loss').toFixed(3)} ` +
      `Reward: ${meanMetrics.get('reward').toFixed(3)} ` +
      `RSS: ${rss.toFixed(3)}mb`);

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

    if (this.options.saveEvery &&
        epochNum % this.options.saveEvery === 0) {
      await this.save(epochNum);
    }
  }

  async run() {
    try {
      await fs.mkdir(this.options.saveDir);
    } catch (e) {
      debug(`Failed to create save dir: ${this.options.saveDir}, ` +
        `error: ${e.stack}`);
    }

    const saveSubDir = path.join(this.options.saveDir, this.options.runName);
    try {
      await fs.mkdir(saveSubDir);
    } catch (e) {
      debug(`Failed to create save subdir: ${saveSubDir}, error: ${e.stack}`);
    }

    for (let i = 0; i < Infinity; i++) {
      await this.runEpoch(i);
    }
  }
}
