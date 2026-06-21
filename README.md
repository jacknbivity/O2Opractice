# O2O 优惠券使用预测 🏪

> **天池新人实战赛 — 生活大实惠：O2O优惠券使用预测**

[![Competition](https://img.shields.io/badge/天池大赛-O2O优惠券预测-blue)](https://tianchi.aliyun.com/competition/entrance/231593/introduction)
[![Python](https://img.shields.io/badge/Python-3.x-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

## 📋 赛题简介

随着移动互联网+O2O（Online to Offline）消费的高速发展，以优惠券盘活老用户或吸引新客户进店消费成为重要营销方式。然而随机投放的优惠券对多数用户造成干扰，个性化投放是提高优惠券核销率的关键技术。

**任务目标**：分析建模，精准预测用户是否会在领取优惠券后的 **15 天内** 使用该优惠券。

---

## 📊 数据说明

| 数据集 | 说明 |
|--------|------|
| `ccf_offline_stage1_train.csv` | 线下消费训练集（用户领券及核销记录） |
| `ccf_offline_stage1_test_revised.csv` | 线下消费测试集 |
| `ccf_online_stage1_train.csv` | 线上消费训练集 |

**字段说明**：

| 字段 | 含义 |
|------|------|
| `user_id` | 用户 ID |
| `merchant_id` | 商户 ID |
| `coupon_id` | 优惠券 ID |
| `discount_rate` | 折扣率（如 `30:20` 表示满30减20） |
| `distance` | 用户与商户距离 |
| `date_received` | 领券日期 |
| `date` | 消费日期（为空表示未消费） |

---

## 🧠 方法概述

### 特征工程

- **时间滑窗划分**：训练窗口 1（4/14–5/14）、训练窗口 2（5/15–6/15）、测试窗口（7月），防止时间穿越
- **用户画像特征**：领券次数、核销率、平均折扣敏感度等
- **商户画像特征**：被领取次数、核销量、优惠力度等
- **用户-商户交互特征**：交叉统计特征
- **折扣相关特征**：满减金额、折扣力度分类

### 模型

- **XGBoost** / **LightGBM** 二分类
- 使用 `StratifiedKFold` 进行交叉验证
- 评估指标：AUC

---

## 🚀 快速开始

### 环境要求

```bash
pip install pandas numpy xgboost lightgbm scikit-learn seaborn matplotlib
```

### 运行

1. 将数据集放入对应目录：
   ```
   ccf_offline_stage1_train/ccf_offline_stage1_train.csv
   ccf_online_stage1_train/ccf_online_stage1_train.csv
   ccf_offline_stage1_test_revised.csv
   ```

2. 运行训练脚本：
   ```bash
   python train.py
   ```

---

## 📁 项目结构

```
.
├── train.py                    # 训练主程序
├── o2o_technical_report.md     # 技术报告（详细方法说明）
├── .gitignore                  # Git 忽略规则
├── .vscode/                    # VS Code 配置
└── README.md                   # 项目说明
```

---

## 📈 结果

详细的技术报告见 [`o2o_technical_report.md`](./o2o_technical_report.md)。

---

## 🔗 参考链接

- [天池大赛 - O2O优惠券使用预测](https://tianchi.aliyun.com/competition/entrance/231593/introduction)
- [天池新人实战赛](https://tianchi.aliyun.com/competition/getstartList)

---

## 📄 License

MIT © [jacknbivity](https://github.com/jacknbivity)
