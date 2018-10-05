var webpack = require('webpack');
var ini = require('ini');
var fs = require('fs');

var path = require('path');
var APP_DIR = path.resolve(__dirname, 'client');
var BUILD_DIR = path.resolve(__dirname, 'server/static/javascript');

var config = {
  entry: APP_DIR + '/main.jsx',
  output: {
    path: BUILD_DIR,
    filename: 'bundle.js'
  },
  externals: {
    'config': JSON.stringify(process.env.NODE_ENV === 'prod' ? ini.parse(fs.readFileSync('./config_files/prod_config.ini', 'utf-8')) : (process.env.NODE_ENV === 'dev' ? ini.parse(fs.readFileSync('./config_files/local_config.ini', 'utf-8')) : '{}'))
  },
  devtool: "#cheap-module-source-map.",  // #eval-source-map" #cheap-module-source-map."
  module: {
    loaders : [
      {
        test : /\.(js|jsx)$/,
        exclude: /node_modules/,
        include : APP_DIR,
        loader : 'babel-loader',
        options: {
            babelrc: false,
            presets:["env", "react", "stage-2"]
        }
      }
    ]
  }
};

module.exports = config;
