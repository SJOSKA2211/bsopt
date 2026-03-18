'use strict';

/**
 * Piscina worker for Gateway
 * Offloads CPU-intensive tasks like deep validation or large object transformation.
 */

const { isMainThread } = require('worker_threads');

async function processData(data) {
  // Placeholder for expensive transformation or validation
  // In a real high-frequency scenario, we might parse large JSON bodies here
  // or perform complex business logic that shouldn't block the main event loop.
  return data;
}

module.exports = async (task) => {
  switch (task.type) {
    case 'PROCESS_DATA':
      return await processData(task.payload);
    default:
      throw new Error(`Unknown task type: ${task.type}`);
  }
};
