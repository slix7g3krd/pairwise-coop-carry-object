# pairwise-coop-carry-object

## 领域

游戏、图形与多媒体

## 主题

双人协作搬运

## 说明

单一约束求解、抓取序号与掉线撤约束。

初始状态下 `gameplay/coop_carry_object.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
