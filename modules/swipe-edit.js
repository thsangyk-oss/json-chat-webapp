'use strict';

module.exports = function register(ctx) {
  return {
    'POST /api/swipe': async (_req, res) => {
      ctx.json(res, 501, {
        error: 'Legacy swipe/edit module is disabled. Use the in-server /api/swipe implementation in server.js.'
      });
    },
    'POST /api/edit-message': async (_req, res) => {
      ctx.json(res, 501, {
        error: 'Legacy swipe/edit module is disabled. Use the in-server /api/edit-message implementation in server.js.'
      });
    },
  };
};
