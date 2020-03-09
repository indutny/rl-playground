import tf from '@tensorflow/tfjs-node';
import Model from './model.js';
import Environment from './env.js';

const LR = 0.001;
const BATCH_SIZE = 1024;
const MAX_STEPS = 128;

async function epoch(epochNum, model, envs, optimizer) {
  for (const env of envs) {
    env.reset();
  }

  const history = [];

  const rewards = new Array(envs.length).fill(0);
  let state = model.initialState;
  for (let t = 0; t < MAX_STEPS; t++) {
    const observation = tf.tensor(envs.map((env) => env.observe()));
    const { probs, newState } = model.forward(observation, state);
    const action = model.sample(probs).dataSync();

    let allFinished = true;
    for (const [ i, env ] of envs.entries()) {
      rewards[i] += env.step(action[i]);
      if (!env.isFinished()) {
        allFinished = false;
      }
    }

    history.push({
      observation,
      state,
      action,
      rewards: rewards.slice(),
    });

    if (allFinished) {
      break;
    }

    state = newState;
  }

  const train = optimizer.minimize(() => {
    let loss = tf.tensor(0);

    for (const { observation, state, action } of history) {
      const { probs } = model.forward(observation, state);

      const sampleMask = tf.oneHot(action, Environment.ACTION_DIMS);
      const logProbs = tf.mul(probs, sampleMask).sum(-1);
      loss = tf.add(loss, logProbs);
    }

    const { rewards } = history[history.length - 1];

    loss = tf.mul(loss.mul(-1), tf.tensor(rewards)).mean(-1);
    return loss;
  }, true);

  let best = Infinity;
  for (const env of envs) {
    if (env.isFinished()) {
      best = Math.min(best, env.steps);
    }
  }

  console.log(`#${epochNum} ` +
    `Loss: ${(await train.dataSync())[0]}, Best: ${best}`);
}

async function main() {
  const optimizer = tf.train.rmsprop(LR);

  const model = new Model(Environment);
  const envs = [];
  for (let i = 0; i < BATCH_SIZE; i++) {
    envs.push(new Environment());
  }

  // Build model
  model.build([ BATCH_SIZE, Environment.OBSERVATION_DIMS ]);

  for (let i = 0; i < Infinity; i++) {
    await epoch(i, model, envs, optimizer);
  }
}

main().catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
