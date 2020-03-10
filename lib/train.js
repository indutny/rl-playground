import path from 'path';
import tf from '@tensorflow/tfjs-node';

import Model from './model.js';
import Environment from './env/maze.js';

const LR = 0.01;
const BATCH_SIZE = 128;
const MAX_STEPS = 128;
const ENTROPY_ALPHA = 0.2;

const SUMMARY_WRITER = tf.node.summaryFileWriter(
  path.join('.', 'logs', process.env.RUN_NAME || 'default'));

async function epoch(epochNum, model, optimizer) {
  const envs = [];
  for (let i = 0; i < BATCH_SIZE; i++) {
    envs.push(new Environment());
  }

  // Tensors to cleanup
  const cleanup = [];

  const history = [];
  const rewards = new Array(envs.length).fill(0);
  let state = model.initialState;
  for (let t = 0; t < MAX_STEPS; t++) {
    const observation = envs.map((env) => env.observe());
    const actionMask = envs.map((env) => env.actionMask());
    const { action: actionTensor, newState, value } = tf.tidy(() => {
      return model.sample(tf.tensor(observation), state, tf.tensor(actionMask));
    });

    const action = await actionTensor.data();
    tf.dispose(actionTensor);
    cleanup.push(newState, value);

    history.push({
      observation,
      actionMask,
      state,
      action,
      rewards: rewards.slice(),
      value: await value.data(),
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

  const train = await optimizer.minimize(() => {
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

      const {
        probs,
        value: valueTensor,
      } = model.forward(tf.tensor(observation), state, tf.tensor(actionMask));

      const sampleMask = tf.oneHot(action, Environment.ACTION_DIMS);
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

    SUMMARY_WRITER.scalar('rl/value_loss', valueLoss.dataSync()[0], epochNum);
    SUMMARY_WRITER.scalar(
      'rl/entropy_loss', entropyLoss.dataSync()[0], epochNum);

    const { rewards } = history[history.length - 1];

    return policyLoss.add(valueLoss).add(entropyLoss.mul(ENTROPY_ALPHA));
  }, true);

  const [ loss ] = await train.data();

  tf.dispose(cleanup);

  let avg = 0;
  let avgCount = 0;
  for (const reward of finalRewards) {
    avg += reward;
    avgCount++;
  }
  avg /= avgCount;

  const rss = process.memoryUsage().rss / 1024 / 1024;

  SUMMARY_WRITER.scalar('rl/avg_reward', avg, epochNum);
  SUMMARY_WRITER.scalar('rl/rss', rss, epochNum);

  console.log(`#${epochNum} Loss: ${loss.toFixed(3)} ` +
    `Average: ${avg.toFixed(3)} RSS: ${rss.toFixed(3)}mb`);

  if (epochNum % 10 === 0) {
    envs[(Math.random() * envs.length) | 0].printTrace();
  }
}

async function main() {
  const optimizer = tf.train.rmsprop(LR);

  const model = new Model(Environment);

  // Build model
  model.build([ BATCH_SIZE, Environment.OBSERVATION_DIMS ]);

  for (let i = 0; i < Infinity; i++) {
    await epoch(i, model, optimizer);
  }
}

main().catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
