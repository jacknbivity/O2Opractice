# O2O优惠券使用预测 - 技术报告

**任务目标**：预测用户在领取优惠券后 15 天内是否使用该券，输出使用概率。

**数据来源**：
- 线下训练集：`ccf_offline_stage1_train/ccf_offline_stage1_train.csv`
- 线下测试集：`ccf_offline_stage1_test_revised.csv`
- 线上训练集：`ccf_online_stage1_train/ccf_online_stage1_train.csv`（当前代码读取但未用于特征提取）

---

## 1. 标签定义与基础处理

**标签定义（get_label）**
- 若 `date` 为空：`label = 0`
- 若 `date_received` 为空：`label = -1`
- 若 `date - date_received <= 15`：`label = 1`
- 其他：`label = 0`

**基础清洗**
- `user_id` / `merchant_id` / `coupon_id` 统一转为字符串。
- `distance` 的字符串 `null` 被替换为缺失值 `NaN`。

---

## 2. 时间滑窗划分

**训练窗口 1**
- 标签窗口（dataset1）：领券日期 `20160414` ~ `20160514`
- 特征窗口（feature1）：`20160101` ~ `20160413` 的历史行为

**训练窗口 2**
- 标签窗口（dataset2）：领券日期 `20160515` ~ `20160615`
- 特征窗口（feature2）：`20160201` ~ `20160514` 的历史行为

**测试窗口**
- 标签窗口（dataset3）：官方测试集（7月领券）
- 特征窗口（feature3）：`20160315` ~ `20160630` 的历史行为

原因：防止时间穿越：特征窗口必须在标签窗口之前，否则会用到“未来信息”，线上评估会大幅掉分。
匹配标签定义：标签是“领券后 15 天内是否核销”，所以标签窗口就是“领券发生的时间段”。特征窗口就是在领券之前可观测到的历史行为。
增加训练样本：用两段连续滑窗（4/14–5/14 与 5/15–6/15）能让模型看到两个不同时间段的用户行为分布，提高泛化能力。

模拟真实测试场景：测试集是 7 月领券，特征使用 3/15–6/30 的历史行为。训练集也采用“历史特征 → 未来领券”的结构，和真实预测一致。
---

## 3. 特征工程总览

> 说明：特征统一在 `get_all_feature` 内组装，所有表通过 `merge` 拼接。以下为函数级别的完整特征列表。

### 3.0 直接保留的原始字段
- `distance`：用户与商家距离，来源于原始字段（字符串 `null` 已转为缺失值）。
- `date_received`：领券日期，既用于时间窗口与泄漏特征，也会保留在模型输入中。

### 3.1 基础与折扣特征（add_discount）
- `if_fd`：是否满减（0/1），`discount_rate` 含 `:` 则为 1。
- `full_value`：满减门槛金额（如 `200:20` 的 200）。
- `reduction_value`：减免金额（如 `200:20` 的 20）。
- `discount_rate`：折扣率（`a:b` 转换为 $1-\frac{b}{a}$，直接折扣保持原值）。

### 3.2 领取日期特征（get_date_feature）
- `day`：领券日（1-31）。
- `day_0.0` ~ `day_6.0`：领券星期几的 one-hot（0 为周一）。

### 3.3 间隔特征（add_day_gap）
- `day_gap`：核销日期与领券日期的天数差（无效日期时为 -1）。

### 3.4 商家特征（get_merchant_feature）
- `merchant_id_sales_use_coupon`：商家优惠券被核销次数。
- `merchant_id_total_coupon`：商家优惠券发放总数。
- `total_coupon`：与 `merchant_id_total_coupon` 同值的冗余列（由赋值产生）。
- `merchant_id_distance_min`：商家被领券且有距离记录的最小距离。
- `merchant_id_distance_max`：商家被领券且有距离记录的最大距离。
- `merchant_id_distance_mean`：商家被领券且有距离记录的平均距离。
- `merchant_id_distance_std`：商家被领券且有距离记录的距离标准差。
- `merchant_id_distance_skew`：商家被领券且有距离记录的距离偏度。
- `first_received_flag`：该条记录是否为该商家最早的领券日期（1/0）。
- `last_received_flag`：该条记录是否为该商家最晚的领券日期（1/0）。
- `is_first_coupon_received`：该商家-优惠券组合的首次领券标记（1/0）。
- `is_last_coupon_received`：该商家-优惠券组合的最后领券标记（1/0）。
- `merchant_coupon_transfer_rate`：商家优惠券核销率 = 使用次数 / 发放次数。

### 3.5 用户特征（get_user_feature）
- `user_id_count_merchant`：用户历史消费过的不同商家数。
- `user_id_distance_min`：用户在优惠券核销行为中的最小距离。
- `user_id_distance_max`：用户在优惠券核销行为中的最大距离。
- `user_id_distance_mean`：用户在优惠券核销行为中的平均距离。
- `user_id_distance_median`：用户在优惠券核销行为中的距离中位数。
- `user_id_distance_std`：用户在优惠券核销行为中的距离标准差。
- `user_id_distance_skew`：用户在优惠券核销行为中的距离偏度。
- `user_id_buy_use_coupon`：用户优惠券核销次数。
- `user_id_coupon_received`：用户优惠券领取总数。
- `user_id_day_gap_min`：用户历史领券到核销的最小天数差。
- `user_id_day_gap_max`：用户历史领券到核销的最大天数差。
- `user_id_day_gap_mean`：用户历史领券到核销的平均天数差。
- `user_id_day_gap_skew`：用户历史领券到核销天数差偏度。
- `first_received_flag`：该条记录是否为该用户最早领券日期（1/0）。
- `last_received_flag`：该条记录是否为该用户最晚领券日期（1/0）。
- `is_first_coupon_received`：该用户-优惠券组合首次领券标记（1/0）。
- `is_last_coupon_received`：该用户-优惠券组合最后领券标记（1/0）。
- `user_coupon_transfer_rate`：用户核销率 = 核销次数 / 领券次数。

### 3.6 用户消费强度特征（user_coupon_consumed）
- `user_consumed_cnt_50_rate`：折扣额 $0<减免\le50$ 的消费占比（再做 Min-Max 归一化）。
- `user_consumed_cnt_200_rate`：折扣额 $50<减免\le200$ 的消费占比（再做 Min-Max 归一化）。
- `user_consumed_cnt_500_rate`：折扣额 $200<减免\le500$ 的消费占比（再做 Min-Max 归一化）。
- `user_consumed_cnt_others_rate`：其他折扣额消费占比（再做 Min-Max 归一化）。

### 3.7 用户-商家交叉特征（get_user_merchant_feature）
- `user_merchant_buy_total`：用户在该商家有消费记录的次数（`date` 非空）。
- `user_merchant_buy_use_coupon`：用户在该商家使用优惠券消费的次数（`date` 与 `date_received` 非空）。
- `user_merchant_any`：用户与该商家的交互次数（含领券与消费）。
- `user_merchant_buy_common`：用户在该商家普通消费次数（`coupon_id == nan`）。
- `user_merchant_coupon_buy_rate`：该商家的用券消费比例 = 用券消费 / 总消费。
- `user_merchant_rate`：用户对该商家的消费比率 = 该商家消费 / 该商家交互。
- `user_merchant_common_buy_rate`：普通消费比例 = 普通消费 / 总消费。

### 3.8 优惠券特征（get_coupon_feature）
- `coupon_type_count`：该优惠券被领取的总次数。
- `coupon_type_used_count`：该优惠券被核销的总次数。
- `coupon_type_used_15_count`：该优惠券在 15 天内被核销的次数。
- `coupon_id_distance_min`：该优惠券核销时的最小距离。
- `coupon_id_distance_max`：该优惠券核销时的最大距离。
- `coupon_id_distance_mean`：该优惠券核销时的平均距离。
- `coupon_id_day_gap_min`：该优惠券领券到核销的最小天数差。
- `coupon_id_day_gap_max`：该优惠券领券到核销的最大天数差。
- `coupon_id_day_gap_mean`：该优惠券领券到核销的平均天数差。
- `coupon_type_used_15_count_rate`：该优惠券 15 天内核销率 = 15 天内核销 / 领取次数。
- `coupon_type_used_count_rate`：该优惠券整体核销率 = 核销 / 领取次数。

### 3.9 用户-优惠券特征（get_user_coupon_feature）
- `user_coupon_type_count`：用户领取该优惠券的次数。
- `user_used_coupon_type_count`：用户核销该优惠券的次数。
- `user_used_coupon_type_count_rate`：用户对该券的核销率 = 核销 / 领取。
- `user_used_coupon_type_used_15_count`：用户在 15 天内核销该券的次数。
- `user_used_coupon_type_used_15_count_rate`：用户对该券的 15 天核销率。
- `user_coupon_day_gap_min`：用户对该券领券到核销的最小天数差。
- `user_coupon_day_gap_max`：用户对该券领券到核销的最大天数差。
- `user_coupon_day_gap_mean`：用户对该券领券到核销的平均天数差。

### 3.10 商家-优惠券特征（get_merchant_coupon_feature）
- `merchant_coupon_type_count`：商家发放该优惠券的次数。
- `merchant_id_used_coupon_type_count`：该优惠券在该商家的核销次数。
- `merchant_id_used_coupon_type_count_rate`：商家对该券的核销率。
- `merchant_used_coupon_type_used_15_count`：该券在该商家 15 天内核销次数。
- `merchant_used_coupon_type_used_15_count_rate`：该券在该商家 15 天内核销率。
- `merchant_coupon_day_gap_min`：该券在该商家领券到核销最小天数差。
- `merchant_coupon_day_gap_max`：该券在该商家领券到核销最大天数差。
- `merchant_coupon_day_gap_mean`：该券在该商家领券到核销平均天数差。

### 3.11 用户-商家-优惠券三方特征（get_user_merchant_coupon_feature）
- `user_merchant_coupon_type_count`：用户在该商家领取该券的次数。
- `user_merchant_id_used_coupon_type_count`：用户在该商家核销该券的次数。
- `user_merchant_id_used_coupon_type_count_rate`：用户-商家-优惠券组合的核销率。
- `user_merchant_used_coupon_type_used_15_count`：用户在该商家 15 天内核销该券的次数。
- `user_merchant_used_coupon_type_used_15_count_rate`：用户在该商家对该券的 15 天核销率。
- `user_merchant_coupon_day_gap_min`：该组合领券到核销的最小天数差。
- `user_merchant_coupon_day_gap_max`：该组合领券到核销的最大天数差。
- `user_merchant_coupon_day_gap_mean`：该组合领券到核销的平均天数差。

### 3.12 时序与泄漏特征（get_leakage_feature）
- `user_coupon_cnt`：用户领券总次数（按用户计数）。
- `user_received_type_coupon`：用户领取该优惠券的次数。
- `coupons_after`：该次领券之后，用户还领了多少张券。
- `coupons_before`：该次领券之前，用户已经领了多少张券。
- `Same_Coupons_Before`：该次之前，用户领过相同优惠券的次数。
- `Same_Coupons_After`：该次之后，用户还领过相同优惠券的次数。
- `days_since_last_coupon`：距上一次领券的天数差。
- `days_until_next_coupon`：距下一次领券的天数差。
- `cupon_count`：用户在该商家领取优惠券的次数（字段原拼写）。
- `Unique_Merchant_Count`：用户领券涉及的不同商家数。
- `Unique_Coupon_Count`：用户领券涉及的不同优惠券种类数。
- `Coupon_Count`：商家被领券的总次数。
- `Specific_Coupon_Count`：商家发放该优惠券的次数。
- `Unique_User_Count`：商家领取优惠券的独立用户数。
- `first_received_flag` / `last_received_flag`：该条记录是否为该用户最早/最晚领券日期。
- `first_received_count` / `last_received_count`：用户在首日/末日领券数量。
- `first_received_flag_mer` / `last_received_flag_mer`：该条记录是否为该商家最早/最晚领券日期。
- `first_received_count_mer` / `last_received_count_mer`：商家在首日/末日领券数量。

---

## 4. 特征归一化与过滤


**过滤与清理**
- 删除全 0 列、全 NaN 列。
- 删除重复行与重复列。
- 训练集删除 `date`、`merchant_id` 字段。

---

## 5. 模型与评估

**模型**：XGBoost（二分类）

- `objective: binary:logistic`

- 5 折 `StratifiedKFold`

- 评估指标为自定义 `myauc`（按 `coupon_id` 分组后求 AUC 再平均）

-  参数： params = {'booster': 'gbtree',

  ​     'objective': 'binary:logistic',

  ​     'eval_metric': 'auc',

  ​     'gamma': 0.1,

  ​     'min_child_weight': 1.1,

  ​     'max_depth': 5,

  ​     'lambda': 10,

  ​     'subsample': 0.7,

  ​     'colsample_bytree': 0.7, 

  ​     'colsample_bylevel': 0.7,

  ​     'eta': 0.01,

  ​     'tree_method': 'exact',

  ​     'seed': 0,

  ​     }

**输出**
- 文件名：`sumbit_xgb<num_boost_round>.csv`
- 内容列：`user_id, coupon_id, date_received, label`
- 输出无表头
