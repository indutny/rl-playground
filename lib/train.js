import tf from '@tensorflow/tfjs-node';
import Model from './model.js';
import Environment from './env.js';

const LR = 0.001;
const BATCH_SIZE = 2048;
const FINISH_DISTANCE = 16;
const MAX_STEPS = 128;

async function epoch(epochNum, model, optimizer) {
  const envs = [];
  for (let i = 0; i < BATCH_SIZE; i++) {
    envs.push(new Environment(FINISH_DISTANCE));
  }

  const history = [];

  const cleanup = [];

  const rewards = new Array(envs.length).fill(0);
  let state = model.initialState;
  for (let t = 0; t < MAX_STEPS; t++) {
    const observation = envs.map((env) => env.observe());
    const { action: actionTensor, newState } = tf.tidy(() => {
      return model.sample(tf.tensor(observation), state);
    });

    const action = await actionTensor.data();
    tf.dispose(actionTensor);
    cleanup.push(newState);

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

  const train = await optimizer.minimize(() => {
    let loss = tf.tensor(0);

    for (const { observation, state, action } of history) {
      const { probs } = model.forward(tf.tensor(observation), state);

      const sampleMask = tf.oneHot(action, Environment.ACTION_DIMS);
      const logProbs = tf.log(tf.mul(probs, sampleMask).sum(-1));
      loss = tf.add(loss, logProbs);
    }

    const { rewards } = history[history.length - 1];

    return tf.mul(loss, tf.tensor(rewards)).mean(-1);
  }, true);

  const [ loss ] = await train.data();

  tf.dispose(cleanup);

  let avg = 0;
  let avgCount = 0;
  for (const [ i, env ] of envs.entries()) {
    if (env.isFinished()) {
      avg += env.steps;
      avgCount++;
    }
  }
  avg /= avgCount;

  const rss = (process.memoryUsage().rss / 1024 / 1024).toFixed(2);
  console.log(`#${epochNum} Loss: ${loss.toFixed(2)} ` +
    `Average: ${avg.toFixed(2)} RSS: ${rss}mb`);

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
