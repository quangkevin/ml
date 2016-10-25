var proxy = require('http-proxy-middleware');

module.exports = {
  server: {
    middleware: {
      1: proxy('/service', {
        target: 'http://localhost:4567',
        changeOrigin: true   // for vhosted sites, changes host header to match to target's host
      }),
      2: require('connect-history-api-fallback')({index: '/index.html', verbose: true})
    }
  }
};
