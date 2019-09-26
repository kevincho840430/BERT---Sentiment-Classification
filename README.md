# BERT - Sentiment Classification
## Topic
- 使用BERT進行中文情意分析
- 使用CNN1D進行預測
## Containing
    1.bert_cnn_training.ipynb
    2.bert_cnn.py
    3.bert_predict.ipynb
    4.data_food.csv
    5.bert.h5
    6.vocab.txt

## Start Training

使用bert_cnn_training.ipynb開始進行訓練
1. 載入dataset(data_food.csv)
2. 進行tokenizer
3. 使用BERT將文字轉換成向量
4. 將轉換出來的向量丟入model訓練
5. 預測結果

## Predict Sentence

使用bert_predict.ipynb進行預測
1. 載入事先訓練好的權重
2. 放入想預測的字並進行轉換向量
3. 預測結果，並設定Threashold判定為好的或壞的
