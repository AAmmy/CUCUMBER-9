# CUCUMBER-9

[MFT2016](http://makezine.jp/event/mft2016/)に出展されていた[CUCUMBER-9（自動きゅうり選果マシーン）](http://makezine.jp/event/makers2016/workpiles/)さんの公開されているデータ https://github.com/workpiles/CUCUMBER-9 を用いて識別.  

### cucu_sgd.py 
http://deeplearning.net/tutorial/ のコードを使用  
epoch 1000 (TITANXで40秒程度) での結果 (test_samples()で実行可能)  

|data|errors|%|
|-----------:|------------:|------------:|
|train|263 / 5376|4.89 %|
|dev|65 / 1512|4.3 %|
|test|57 / 1512|3.77 %|

### cucu_mlp.py 
http://deeplearning.net/tutorial/ のコードを使用  
epoch 8000  

|data|errors|%|
|-----------:|------------:|------------:|
|train|373 / 5376|6.94 %|
|dev|104 / 1512|6.88 %|
|test|97 / 1512|6.42 %|
