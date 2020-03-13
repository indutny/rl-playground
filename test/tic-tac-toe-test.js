import assert from 'assert';

import { TicTacToe } from '../lib/env/index.js';

describe('TicTacToe', () => {
  it('should compute winner', async () => {
    const env = new TicTacToe();

    assert.deepStrictEqual(await env.observe(), [
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      1,
    ]);
    assert.deepStrictEqual(env.actionMask(), [
      1, 1, 1,
      1, 1, 1,
      1, 1, 1,
      0,
    ]);
    assert.strictEqual(await env.step(0), 0);

    assert.deepStrictEqual(await env.observe(), [
      -1, 0, 0,
      0, 0, 0,
      0, 0, 0,
      -1,
    ]);
    assert.deepStrictEqual(env.actionMask(), [
      0, 1, 1,
      1, 1, 1,
      1, 1, 1,
      0,
    ]);
    assert.strictEqual(await env.step(1), 0);

    assert.deepStrictEqual(await env.observe(), [
      1, -1, 0,
      0, 0, 0,
      0, 0, 0,
      1,
    ]);
    assert.deepStrictEqual(env.actionMask(), [
      0, 0, 1,
      1, 1, 1,
      1, 1, 1,
      0,
    ]);
    assert.strictEqual(await env.step(4), 0);

    assert.deepStrictEqual(await env.observe(), [
      -1, 1, 0,
      0, -1, 0,
      0, 0, 0,
      -1,
    ]);
    assert.deepStrictEqual(env.actionMask(), [
      0, 0, 1,
      1, 0, 1,
      1, 1, 1,
      0,
    ]);
    assert.strictEqual(await env.step(2), 0);
    assert.ok(!env.isFinished());

    assert.deepStrictEqual(await env.observe(), [
      1, -1, -1,
      0, 1, 0,
      0, 0, 0,
      1,
    ]);
    assert.deepStrictEqual(env.actionMask(), [
      0, 0, 0,
      1, 0, 1,
      1, 1, 1,
      0,
    ]);
    assert.strictEqual(await env.step(8), 1);

    assert.ok(!env.isFinished());

    assert.deepStrictEqual(env.actionMask(), [
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      1,
    ]);
    assert.strictEqual(await env.step(9), -1);
    assert.ok(env.isFinished());

    assert.strictEqual(await env.step(9), 0);
  });
});
