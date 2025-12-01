# Are Transformers Effective for Time Series Forecasting?


In this work we showed that despite the recent popularity of LSTM in time series forecasting (TSF) they do not appear to meaningfully improve performance. A simple baseline, "PAttn" was proposed, which outperformed most LLM-based TSF models, Linear models and LTSM base models. 

Authors: Kun Qian, Zhuo Qian, Yanke Li 

## Overview 💁🏼
Recent work in time series analysis has increasingly focused on adapting transformer for **forecasting (TSF)**, classification, and anomaly detection. These studies suggest that transformer models, designed for sequential dependencies in text, could generalize to time series data. While this idea aligns with the popularity of transformer models in machine learning, direct connections between transformer modeling and TSF remain unclear. **How beneficial are transformer models for traditional TSF task?**

Through a series of studies on three recent **transformer-based TSF** methods, we found that transformer model led to improvements. Additionally, we introduced PAttn, showing that patching and attention structures can perform well.

![Ablations/PAttn](pic/transformer.png)

## Dataset 📖
You can access the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view), then place the downloaded contents under ./datasets

## Setup 🔧
Three different popular groups of methods were included in our ablation approach. one group is w/o transformer, another group is w/ transformer, the other group is LLM based.

##  w/o transformer
### Linear
A basic linear baseline that captures simple global trends but fails on nonlinear or long-range temporal dependencies.

     cd ./Linear
     bash ./scripts/ETTh.sh (for ETTh1 & ETTh2)
     bash ./scripts/ETTm.sh (for ETTm1 & ETTm2)
     bash ./scripts/weather.sh (for Weather)
### DLinear
Improves linear forecasting by decomposing trend and residual components, offering greater robustness to non-stationarity.

     cd ./DLinear
     bash ./scripts/ETTh.sh (for ETTh1 & ETTh2)
     bash ./scripts/ETTm.sh (for ETTm1 & ETTm2)
     bash ./scripts/weather.sh (for Weather)
### NLinear
Stabilizes linear forecasting through normalization, helping models perform better on datasets with large scale variations.

     cd ./NLinear
     bash ./scripts/ETTh.sh (for ETTh1 & ETTh2)
     bash ./scripts/ETTm.sh (for ETTm1 & ETTm2)
     bash ./scripts/weather.sh (for Weather)
### LSTM
A recurrent model capable of learning nonlinear temporal patterns but limited by vanishing gradients and long-sequence inefficiency.

     cd ./LSTM
     bash ./scripts/ETTh.sh (for ETTh1 & ETTh2)
     bash ./scripts/ETTm.sh (for ETTm1 & ETTm2)
     bash ./scripts/weather.sh (for Weather)
### p-sLSTM
Enhances traditional LSTMs by selectively controlling state updates, enabling better long-term dependency modeling with higher efficiency.

     cd ./P-sLSTM
     bash ./scripts/EXP-LongForecasting/P_sLSTM/ETTh.sh (for ETTh1 & ETTh2)
     bash ./scripts/EXP-LongForecasting/P_sLSTM/ETTm.sh (for ETTm1 & ETTm2)
     bash ./scripts/EXP-LongForecasting/P_sLSTM/weather.sh (for Weather)

#### For other datasets, Please change the script name in above command.

## w/ transformer

### iTransformer
A variable-wise Transformer that improves long-sequence forecasting by modeling channels independently rather than time steps.

     cd ./PAttn
     bash ./scripts/ETTh.sh (for ETTh1 & ETTh2)
     bash ./scripts/ETTm.sh (for ETTm1 & ETTm2)
     bash ./scripts/weather.sh (for Weather)

### PatchTST
Uses patch embeddings to capture local temporal patterns, achieving state-of-the-art long-horizon forecasting performance.

     cd ./PAttn
     bash ./scripts/ETTh.sh (for ETTh1 & ETTh2)
     bash ./scripts/ETTm.sh (for ETTm1 & ETTm2)
     bash ./scripts/weather.sh (for Weather)

### PAttn
A lightweight attention mechanism that reduces computational cost while retaining essential temporal relationships for accurate forecasting.

     cd ./PAttn
     bash ./scripts/ETTh.sh (for ETTh1 & ETTh2)
     bash ./scripts/ETTm.sh (for ETTm1 & ETTm2)
     bash ./scripts/weather.sh (for Weather)

#### For other datasets, Please change the script name in above command.

## LLM
#### Run on CALF (ETT) :
     
    cd ./CALF
    sh scripts/long_term_forecasting/ETTh_GPT2.sh
    sh scripts/long_term_forecasting/ETTm_GPT2.sh
    
    sh scripts/long_term_forecasting/traffic.sh 
    (For other datasets, such as traffic)

#### Run on OneFitsAll (ETT) :
     cd ./OFA
     bash ./script/ETTh_GPT2.sh   
     bash ./script/ETTm_GPT2.sh

     bash ./script/illness.sh 
     (For other datasets, such as illness)

#### Run on Time-LLM (ETT) 
     cd ./Time-LLM-exp
     bash ./scripts/train_script/TimeLLM_ETTh1.sh
     bash ./scripts/train_script/TimeLLM_ETTm1.sh 

     bash ./scripts/train_script/TimeLLM_Weather.sh
     (For other datasets, such as Weather)

#### (To run on other datasets, please change the dataset name as shown in example.)

## Acknowledgement

This codebase is built based on the [LLMsForTimeSeries](https://github.com/BennyTMT/LLMsForTimeSeries). Thanks!


