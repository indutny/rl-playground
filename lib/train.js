import tf from '@tensorflow/tfjs-node';
import Model from './model.js';
import Environment from './env/maze.js';

const LR = 0.01;
const BATCH_SIZE = 256;
const MAX_STEPS = 128;

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
    const { action: actionTensor, newState } = tf.tidy(() => {
      return model.sample(tf.tensor(observation), state, tf.tensor(actionMask));
    });

    const action = await actionTensor.data();
    tf.dispose(actionTensor);
    cleanup.push(newState);

    history.push({
      observation,
      actionMask,
      state,
      action,
      rewards: rewards.slice(),
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
    let loss = tf.tensor(0);

    for (const { observation, actionMask, state, action, rewards } of history) {
      const { probs } =
        model.forward(tf.tensor(observation), state, tf.tensor(actionMask));

      const sampleMask = tf.oneHot(action, Environment.ACTION_DIMS);
      const logProbs = tf.mul(probs, sampleMask).sum(-1);

      const rewardToGo = finalRewards.slice();
      for (let i = 0; i < rewards.length; i++) {
        rewardToGo[i] -= rewards[i];
      }
      loss = tf.add(loss, logProbs.mul(tf.tensor(rewardToGo)).mean(-1));
    }

    const { rewards } = history[history.length - 1];

    // This is how the gradients should be!
    loss = loss.mul(-1);
    return loss;
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
