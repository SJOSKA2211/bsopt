'use strict';

/**
 * Piscina worker for Gateway
 * Offloads CPU-intensive tasks like deep validation or large object transformation.
 */

const { isMainThread } = require('worker_threads');

async function processData(data) {
  const toSnakeCase = (str) => str.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
  
  const transform = (obj) => {
    if (Array.isArray(obj)) return obj.map(transform);
    if (obj !== null && typeof obj === 'object') {
      return Object.entries(obj).reduce((acc, [key, value]) => {
        acc[toSnakeCase(key)] = transform(value);
        return acc;
      }, {});
    }
    return obj;
  };

  return transform(data);
}

module.exports = async (task) => {
  switch (task.type) {
    case 'PROCESS_DATA':
      return await processData(task.payload);
    default:
      throw new Error(`Unknown task type: ${task.type}`);
  }
};
