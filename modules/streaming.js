module.exports = function register(ctx) {
  return {
    'POST /api/chat-stream': async (req, res) => {
      res.writeHead(501, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({
        error: 'Legacy streaming module is disabled. Use the in-server /api/chat-stream implementation in server.js.'
      }));
    },
  };
};
