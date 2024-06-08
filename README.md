# Attack-TDM

## 运行方式
```
pip install -r requirements.txt
python attack.py
```

```
options:
    --target_class 攻击类别 <cat / car / ...>
    --gpt 使用gpt-4产生候选词
```

生成的图片位于同级目录下`./figures/<gpt / tokenizer>/<target_class>`