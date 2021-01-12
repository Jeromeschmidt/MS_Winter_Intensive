# QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
# Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import clr
clr.AddReference("System")
clr.AddReference("QuantConnect.Algorithm")
clr.AddReference("QuantConnect.Common")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *

import json
import numpy as np
import pandas as pd
from io import StringIO
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, SimpleRNN
from keras.optimizers import SGD, Adam
from keras.utils.generic_utils import serialize_keras_object

class KerasNeuralNetworkAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 8, 1)   # Set Start Date
        self.SetEndDate(2020, 8, 31)     # Set End Date
        self.SetCash(100000)            # Set Strategy Cash
        self.SetBenchmark("SPY")

        self.modelBySymbol = {}
        
        self.tickers = ["SPY", "QQQ", "TLT", "PYPL", "AAPL", "YWLO", "MSFT", "AMZN", "NOW", "MA", "TDG", "ADSK", "CHTR", "V", "GOOG", "DIS", "FB", "NFLX", "UBER", "SHOP", "BKNG", "SCHW", "FIS"]

        for ticker in self.tickers:
            symbol = self.AddEquity(ticker).Symbol
            # Read the model saved in the ObjectStore
            if self.ObjectStore.ContainsKey(f'{symbol}_model'):
                modelStr = self.ObjectStore.Read(f'{symbol}_model')
                config = json.loads(modelStr)['config']
                self.modelBySymbol[symbol] = Sequential.from_config(config)
                self.Debug(f'Model for {symbol} sucessfully retrieved from the ObjectStore')

        # Look-back period for training set
        self.lookback = 30
        
        # Max Drawdown
        self.maximumDrawdownPercent = 0.02
        
        # Max Position Size
        self.maximumPositionSize = 0.2

        # # Train Neural Network every day
        # self.Train(
        #     self.DateRules.EveryDay(),
        #     self.TimeRules.At(8, 0),
        #     self.NeuralNetworkTraining)
        
        # Train Neural Network every monday
        self.Train(
            self.DateRules.Every(DayOfWeek.Monday),
            self.TimeRules.AfterMarketOpen("SPY"),
            self.NeuralNetworkTraining)


        # # Place trades 30 minutes after the market is open
        # self.Schedule.On(
        #     self.DateRules.EveryDay("SPY"),
        #     self.TimeRules.AfterMarketOpen("SPY", 30),
        #     self.Trade)
        
        # Place trades 30 minutes after the market is open every Monday
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.Trade)
            
        # # Liquidate Porfolio before market close each day
        # self.Schedule.On(
        #     self.DateRules.EveryDay("SPY"),
        #     self.TimeRules.BeforeMarketClose("SPY", 5),
        #     self.Liquidate())


    def OnEndOfAlgorithm(self):
        ''' Save the data and the mode using the ObjectStore '''
        for symbol, model in self.modelBySymbol.items():
            modelStr = json.dumps(serialize_keras_object(model))
            self.ObjectStore.Save(f'{symbol}_model', modelStr)
            self.Debug(f'Model for {symbol} sucessfully saved in the ObjectStore')


    def NeuralNetworkTraining(self):
        '''Train the Neural Network and save the model in the ObjectStore'''
        symbols = self.Securities.keys()

        # Minute historical data is used to train the machine learning model
        # history = self.History(symbols, self.lookback + 1, Resolution.Minute)
        history = self.History(symbols, self.lookback + 1, Resolution.Daily)
        history = history.open.unstack(0)

        for symbol in symbols:
            if symbol not in history:
                continue

            predictor = history[symbol][:-1]
            predictand = history[symbol][1:]

            # build a neural network from the 1st layer to the last layer
            model = Sequential()

            # model.add(Dense(10, input_dim = 1))
            # model.add(Activation('relu'))
            # model.add(LSTM(units = 10, return_sequences=True))
            # model.add(Dense(1))

            # sgd = SGD(lr = 0.01)   # learning rate = 0.01

            # # choose loss function and optimizing method
            # model.compile(loss='mse', optimizer=sgd)
            
            # # Add our first LSTM layer - 50 nodes
            # model.add(LSTM(units = 50, return_sequences=True, input_shape=(history[symbol][0])))
            # # Add Dropout layer to avoid overfitting
            # model.add(Dropout(0.2))
            # # Add additional layers
            # model.add(LSTM(units=50, return_sequences=True))
            # model.add(Dropout(0.2))
            # model.add(LSTM(units=50, return_sequences=True))
            # model.add(Dropout(0.2))
            # model.add(LSTM(units=50))
            # model.add(Dropout(0.2))
            # model.add(Dense(units = 1))

            # # Compile the model
            # model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae', 'acc'])
            
            
            model.add(Dense(128, input_dim = 1))
            model.add(Activation('relu'))
            model.add(Dense(64, input_dim = 1))
            model.add(Activation('relu'))
            model.add(Dense(32, input_dim = 1))
            model.add(Activation('relu'))
            model.add(Dense(16, input_dim = 1))
            model.add(Activation('relu'))
            model.add(Dense(8, input_dim = 1))
            model.add(Activation('relu'))
            model.add(Dense(1))

            sgd = SGD(lr = 0.01)   # learning rate = 0.01

            # choose loss function and optimizing method
            model.compile(loss='mse', optimizer=sgd)
            
            # model.add(SimpleRNN(10, return_sequences=True, return_state=True))
            # model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae', 'acc'])
            
            # pick an iteration number large enough for convergence
            for step in range(200):
                # training the model
                cost = model.train_on_batch(predictor, predictand)

            self.modelBySymbol[symbol] = model


    def Trade(self):
        '''
        Predict the price using the trained model and out-of-sample data
        Enter or exit positions based on relationship of the open price of the current bar and the prices defined by the machine learning model.
        Liquidate if the open price is below the sell price and buy if the open price is above the buy price
        '''
        target = 1 / len(self.Securities)
        
        predicted_up = list()
        predicted_down = list()
        
        for symbol, model in self.modelBySymbol.items():

            # Get the out-of-sample history
            history = self.History(symbol, self.lookback, Resolution.Daily)
            history = history.open.unstack(0)[symbol]

            # Get the final predicted price
            prediction = model.predict(history)[0][-1]
            historyStd = np.std(history)

            holding = self.Portfolio[symbol]
            openPrice = self.CurrentSlice[symbol].Open
            
            if openPrice < prediction - historyStd:
                predicted_down.append(symbol)
            else:
                predicted_up.append(symbol)
            
            position_size =  (1/len(predicted_up))# if  (1/len(predicted_up)) < self.maximumPositionSize else self.maximumPositionSize
            
            targets = list()
            
            for ticker in predicted_down:
                # targets.append(PortfolioTarget.Percent(algorithm, insight.Symbol, 0))
                self.SetHoldings(ticker, 0)
                
            for ticker in predicted_up:
                # targets.append(PortfolioTarget.Percent(algorithm, insight.Symbol, position_size))
                # self.SetHoldings(symbol, position_size)
                targets.append(PortfolioTarget(ticker, position_size))
            
            self.SetHoldings(targets)
            
            ### Risk Management
            for ticker in self.tickers:
                if self.Portfolio[ticker].Invested:
                    if self.Portfolio[ticker].UnrealizedProfit > self.maximumDrawdownPercent:
                        self.SetHoldings(ticker, 0)
