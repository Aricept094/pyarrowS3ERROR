#Optuna_test



import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import talib as ta
import pandas_ta as taa
import optuna
import eli5
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.search.optuna import OptunaSearch
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, LeaveOneOut, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from imblearn.metrics import geometric_mean_score
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from ta import add_all_ta_features
from eli5.sklearn import PermutationImportance
import ta as tax
import ray

df = pd.read_csv('C:/Work/AI/trade/merged_file.csv')

# Create a new column in the DataFrame to hold the trading signal
df['Signal'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

df = add_all_ta_features(df, "open", "high", "low", "close", "volume", fillna=True)
df['AD'] = ta.AD(df['high'], df['low'], df['close'], df['volume'])
df['ADOSC'] = ta.ADOSC(df['high'], df['low'], df['close'], df['volume'])
df['ADXR'] = ta.ADXR(df['high'], df['low'], df['close'])
df['APO'] = ta.APO(df['adjusted close'])
df['AROON_UP'], df['AROON_DOWN'] = ta.AROON(df['high'], df['low'])
df['AROONOSC'] = ta.AROONOSC(df['high'], df['low'])
df['AVGPRICE'] = ta.AVGPRICE(df['open'], df['high'], df['low'], df['close'])
df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = ta.BBANDS(df['adjusted close'])
df['BETA'] = ta.BETA(df['high'], df['low'])
df['BOP'] = ta.BOP(df['open'], df['high'], df['low'], df['close'])
df['CDL2CROWS'] = ta.CDL2CROWS(df['open'], df['high'], df['low'], df['close'])
df['CDL3BLACKCROWS'] = ta.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
df['CDL3INSIDE'] = ta.CDL3INSIDE(df['open'], df['high'], df['low'], df['close'])
df['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(df['open'], df['high'], df['low'], df['close'])
df['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(df['open'], df['high'], df['low'], df['close'])
df['CDL3STARSINSOUTH'] = ta.CDL3STARSINSOUTH(df['open'], df['high'], df['low'], df['close'])
df['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
df['CDLABANDONEDBABY'] = ta.CDLABANDONEDBABY(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLADVANCEBLOCK'] = ta.CDLADVANCEBLOCK(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLBELTHOLD'] = ta.CDLBELTHOLD(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLBREAKAWAY'] = ta.CDLBREAKAWAY(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLCLOSINGMARUBOZU'] = ta.CDLCLOSINGMARUBOZU(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLCONCEALBABYSWALL'] = ta.CDLCONCEALBABYSWALL(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLCOUNTERATTACK'] = ta.CDLCOUNTERATTACK(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLDOJI'] = ta.CDLDOJI(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLDOJISTAR'] = ta.CDLDOJISTAR(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLENGULFING'] = ta.CDLENGULFING(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLGAPSIDESIDEWHITE'] = ta.CDLGAPSIDESIDEWHITE(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLHAMMER'] = ta.CDLHAMMER(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLHARAMI'] = ta.CDLHARAMI(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLHARAMICROSS'] = ta.CDLHARAMICROSS(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLHIGHWAVE'] = ta.CDLHIGHWAVE(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLHIKKAKE'] = ta.CDLHIKKAKE(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLHIKKAKEMOD'] = ta.CDLHIKKAKEMOD(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLHOMINGPIGEON'] = ta.CDLHOMINGPIGEON(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLIDENTICAL3CROWS'] = ta.CDLIDENTICAL3CROWS(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLINNECK'] = ta.CDLINNECK(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLKICKING'] = ta.CDLKICKING(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLKICKINGBYLENGTH'] = ta.CDLKICKINGBYLENGTH(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLLADDERBOTTOM'] = ta.CDLLADDERBOTTOM(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLLONGLEGGEDDOJI'] = ta.CDLLONGLEGGEDDOJI(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLLONGLINE'] = ta.CDLLONGLINE(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLMARUBOZU'] = ta.CDLMARUBOZU(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLMATCHINGLOW'] = ta.CDLMATCHINGLOW(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLMATHOLD'] = ta.CDLMATHOLD(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLMORNINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLONNECK'] = ta.CDLONNECK(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLPIERCING'] = ta.CDLPIERCING(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLRICKSHAWMAN'] = ta.CDLRICKSHAWMAN(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLRISEFALL3METHODS'] = ta.CDLRISEFALL3METHODS(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLSEPARATINGLINES'] = ta.CDLSEPARATINGLINES(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLSHORTLINE'] = ta.CDLSHORTLINE(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLSTALLEDPATTERN'] = ta.CDLSTALLEDPATTERN(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLSTICKSANDWICH'] = ta.CDLSTICKSANDWICH(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLTAKURI'] = ta.CDLTAKURI(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLTASUKIGAP'] = ta.CDLTASUKIGAP(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLTHRUSTING'] = ta.CDLTHRUSTING(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLTRISTAR'] = ta.CDLTRISTAR(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLUNIQUE3RIVER'] = ta.CDLUNIQUE3RIVER(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLUPSIDEGAP2CROWS'] = ta.CDLUPSIDEGAP2CROWS(df['open'], df['high'], df['low'], df['adjusted close'])
df['CDLXSIDEGAP3METHODS'] = ta.CDLXSIDEGAP3METHODS(df['open'], df['high'], df['low'], df['adjusted close'])
df['CMO'] = ta.CMO(df['adjusted close'])
df['CORREL'] = ta.CORREL(df['high'], df['low'])
df['DEMA'] = ta.DEMA(df['adjusted close'])
df['DX'] = ta.DX(df['high'], df['low'], df['adjusted close'])
df['HT_DCPERIOD'] = ta.HT_DCPERIOD(df['adjusted close'])
df['HT_DCPHASE'] = ta.HT_DCPHASE(df['adjusted close'])
inphase_ht, quadrature_ht = ta.HT_PHASOR(df['adjusted close'])
df['HT_PHASOR_INPHASE'] = pd.Series(inphase_ht)
df['HT_PHASOR_QUADRATURE'] = pd.Series(quadrature_ht)
sine_ht, leadsine_ht = ta.HT_SINE(df['adjusted close'])
df['HT_SINE'] = pd.Series(sine_ht)
df['HT_LEADSINE'] = pd.Series(leadsine_ht)
df['HT_TRENDLINE'] = ta.HT_TRENDLINE(df['adjusted close'])
df['HT_TRENDMODE'] = ta.HT_TRENDMODE(df['adjusted close'])
df['LINEARREG'] = ta.LINEARREG(df['adjusted close'])
df['LINEARREG_ANGLE'] = ta.LINEARREG_ANGLE(df['adjusted close'])
df['LINEARREG_INTERCEPT'] = ta.LINEARREG_INTERCEPT(df['adjusted close'])
df['LINEARREG_SLOPE'] = ta.LINEARREG_SLOPE(df['adjusted close'])
df['MA'] = ta.MA(df['adjusted close'])
macd_ext, macd_signal_ext, macd_histogram_ext = ta.MACDEXT(df['adjusted close'])
df['MACD_EXT'] = pd.Series(macd_ext)
df['MACD_SIGNAL_EXT'] = pd.Series(macd_signal_ext)
df['MACD_HISTOGRAM_EXT'] = pd.Series(macd_histogram_ext)
macd_fix, macd_signal_fix, macd_histogram_fix = ta.MACDFIX(df['adjusted close'])
df['MACD_FIX'] = pd.Series(macd_fix)
df['MACD_SIGNAL_FIX'] = pd.Series(macd_signal_fix)
df['MACD_HISTOGRAM_FIX'] = pd.Series(macd_histogram_fix)
df['MAMA'], df['FAMA'] = ta.MAMA(df['adjusted close'])
df['MAX'] = ta.MAX(df['adjusted close'])
df['MAXINDEX'] = ta.MAXINDEX(df['adjusted close'])
df['MEDPRICE'] = ta.MEDPRICE(df['high'], df['low'])
df['MIDPOINT'] = ta.MIDPOINT(df['adjusted close'])
df['MIDPRICE'] = ta.MIDPRICE(df['high'], df['low'])
df['MININDEX'] = ta.MININDEX(df['low'])
df['MINUS_DI'] = ta.MINUS_DI(df['high'], df['low'], df['adjusted close'])
df['MINUS_DM'] = ta.MINUS_DM(df['high'], df['low'])
df['MOM'] = ta.MOM(df['adjusted close'])
df['NATR'] = ta.NATR(df['high'], df['low'], df['adjusted close'])
df['PLUS_DI'] = ta.PLUS_DI(df['high'], df['low'], df['adjusted close'])
df['PLUS_DM'] = ta.PLUS_DM(df['high'], df['low'])
df['ROCP'] = ta.ROCP(df['adjusted close'])
df['ROCR'] = ta.ROCR(df['adjusted close'])
df['ROCR100'] = ta.ROCR100(df['adjusted close'])
df['SAR'] = ta.SAR(df['high'], df['low'])
df['SAREXT'] = ta.SAREXT(df['high'], df['low'])
df['STDDEV'] = ta.STDDEV(df['adjusted close'])
df['STOCH_K'], df['STOCH_D'] = ta.STOCH(df['high'], df['low'], df['adjusted close'])
df['STOCHF_K'], df['STOCHF_D'] = ta.STOCHF(df['high'], df['low'], df['adjusted close'])
df['STOCHRSI_K'], df['STOCHRSI_D'] = ta.STOCHRSI(df['adjusted close'])
df['SUM'] = ta.SUM(df['adjusted close'])
df['T3'] = ta.T3(df['adjusted close'])
df['TEMA'] = ta.TEMA(df['adjusted close'])
df['TRANGE'] = ta.TRANGE(df['high'], df['low'], df['adjusted close'])
df['TRIMA'] = ta.TRIMA(df['adjusted close'])
df['TSF'] = ta.TSF(df['adjusted close'])
df['TYPPRICE'] = ta.TYPPRICE(df['high'], df['low'], df['adjusted close'])
df['VAR'] = ta.VAR(df['adjusted close'])
df['WCLPRICE'] = ta.WCLPRICE(df['high'], df['low'], df['adjusted close'])
df['bullish_engulfing'] = (df['open'] > df['close'].shift(1)) & (df['close'] > df['open'].shift(1)) & (df['open'].shift(1) > df['close']) & (df['close'].shift(1) > df['open'])
df['bearish_engulfing'] = (df['open'] < df['close'].shift(1)) & (df['close'] < df['open'].shift(1)) & (df['open'].shift(1) < df['close']) & (df['close'].shift(1) < df['open'])
df["bullish_engulfing"] = df["bullish_engulfing"].astype(int)
df["bearish_engulfing"] = df["bearish_engulfing"].astype(int)
df["CDLBELTHOLD"] = df["CDLBELTHOLD"].replace({100: 1, -100: -1})
df["CDLCLOSINGMARUBOZU"] = df["CDLCLOSINGMARUBOZU"].replace({100: 1, -100: -1})
df["CDLDOJI"] = df["CDLDOJI"].replace({100: 1, -100: -1})
df["CDLENGULFING"] = df["CDLENGULFING"].replace({100: 1, -100: -1})
df["CDLHARAMI"] = df["CDLHARAMI"].replace({100: 1, -100: -1})
df["CDLHIGHWAVE"] = df["CDLHIGHWAVE"].replace({100: 1, -100: -1})
df["CDLHIKKAKE"] = df["CDLHIKKAKE"].replace({100: 1, -100: -1,200: 1, -200: -1})
df["CDLLONGLEGGEDDOJI"] = df["CDLLONGLEGGEDDOJI"].replace({100: 1, -100: -1})
df["CDLLONGLINE"] = df["CDLLONGLINE"].replace({100: 1, -100: -1})
df["CDLMARUBOZU"] = df["CDLMARUBOZU"].replace({100: 1, -100: -1})
df["CDLRICKSHAWMAN"] = df["CDLRICKSHAWMAN"].replace({100: 1, -100: -1})
df["CDLSHORTLINE"] = df["CDLSHORTLINE"].replace({100: 1, -100: -1})
df["CDLSPINNINGTOP"] = df["CDLSPINNINGTOP"].replace({100: 1, -100: -1})

missing_values_pct = df.isnull().mean() * 100
cols_to_drop = missing_values_pct[missing_values_pct > 85].index
df.drop(cols_to_drop, axis=1, inplace=True)
mode_values = df.mode().iloc[0]
constant_value_percentage = (df == mode_values).mean()
df.drop(columns=df.columns[constant_value_percentage > 0.95], inplace=True)
df.dropna(inplace=True)
X = df.drop(['time', 'Signal'], axis=1)
y = df['Signal']


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=88)
for train_val_index, test_index in sss.split(X, y):
    X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
    y_train_val, y_test = y.iloc[train_val_index], y.iloc[test_index]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=88)
for train_index, val_index in sss.split(X_train_val, y_train_val):
    X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
    y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

ray.init(_metrics_export_port=9191)
input_size = X_train.shape[1]
num_cores = 16

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size,  hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1, input_size)
        h_0, c_0 = self.init_hidden(x.shape[0], x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.linear(out[:, -1])
        return out.squeeze()

    def init_hidden(self, batch_size, device):
        h_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        c_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        return h_0, c_0

def train_model(model, optimizer, criterion, data_loader, device, scaler, scheduler):
    model.train()
    total_loss = 0
    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = model(x_batch)
            loss = criterion(output, y_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # to check for any possible inf/nan gradients

        # step the optimizer manually
        optimizer.step()

        # update the scaler
        scaler.update()

        # step the scheduler after the optimizer
        scheduler.step()

        total_loss += loss.item()

    return total_loss



def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            predictions.extend(torch.sigmoid(output).detach().cpu().numpy().flatten())
    return predictions

def objective(trial, device):
    
    hidden_size = trial.suggest_int('hidden_size', 500, 2000)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1.0, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD', 'AdamW'])
    batch_size = trial.suggest_int('batch_size', 32, 256)
    scheduler_name = trial.suggest_categorical('lr_scheduler', ['StepLR', 'ExponentialLR'])
    gamma = trial.suggest_float('gamma', 0.05, 1.0)
    step_size = trial.suggest_int('step_size', 1, 100)

    model = LSTMModel(input_size, hidden_size, 1, num_layers, dropout)
    model.to(device)
    
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.MSELoss()

    optimizer_classes = {
        'Adam': torch.optim.Adam,
        'RMSprop': torch.optim.RMSprop,
        'SGD': torch.optim.SGD,
        'AdamW': torch.optim.AdamW
    }
    optimizer = optimizer_classes[optimizer_name](model.parameters(), lr=lr)

    scheduler_classes = {
        'StepLR': torch.optim.lr_scheduler.StepLR,
        'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
    }

    if scheduler_name == 'StepLR':
        scheduler = scheduler_classes[scheduler_name](optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'ExponentialLR':
        scheduler = scheduler_classes[scheduler_name](optimizer, gamma=gamma)

    train_data_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, pin_memory=True)
    val_data_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, pin_memory=True)

    for epoch in range(40):
        train_loss = train_model(model, optimizer, criterion, train_data_loader, device, scaler, scheduler)
        intermediate_value = 1.0 / (train_loss + 1e-5)
        trial.report(intermediate_value, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    predictions_val = evaluate_model(model, val_data_loader, device)
    binary_predictions_val = (np.array(predictions_val) > 0.5).astype(int)
    binary_labels_val = y_val.numpy().reshape(-1)
    f1_val = f1_score(binary_labels_val, binary_predictions_val)

    trial.set_user_attr("f1_val", f1_val)

    return f1_val

def trainable(config, checkpoint_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trial = optuna.trial.FixedTrial(config)
    result = objective(trial, device)
    tune.report(score=result)
                  
if __name__ == "__main__":
    resources_per_trial = {"gpu": 1, "cpu": num_cores} if torch.cuda.is_available() else {"cpu": num_cores}
    scheduler = MedianStoppingRule(metric="score", mode="max")
    search_alg = OptunaSearch(metric="score", mode="max")

    analysis = tune.run(
        trainable,
        config={
            "input_size": input_size,
            "hidden_size": tune.randint(500, 2000),
            "num_layers": tune.randint(1, 5),
            "dropout": tune.uniform(0.0, 0.5),
            "lr": tune.loguniform(1e-5, 1.0),
            "optimizer": tune.choice(['Adam', 'RMSprop', 'SGD', 'AdamW']),
            "batch_size": tune.randint(32, 256),
            "lr_scheduler": tune.choice(['StepLR', 'ExponentialLR']),
            "gamma": tune.uniform(0.05, 1.0),
            "step_size": tune.randint(1, 100),
        },
        resources_per_trial=resources_per_trial,
        num_samples=15,
        scheduler=scheduler,
        search_alg=search_alg,
    )

    best_parameters = analysis.get_best_config(metric="score", mode="max")
    best_trial = analysis.get_best_trial(metric="score", mode="max")
    
    print('Best Trial: score {},\nparams {}'.format(best_trial.last_result["score"], best_parameters))

    for trial in analysis.trials:
        print(f"Trial {trial.trial_id}, F1 score: {trial.last_result['score']}")
        
    ray.shutdown()


