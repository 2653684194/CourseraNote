```
EX14_Transformer/
├── 📄 transformer_comparison.py        # 主文件（包含A/B两个版本）
├── 📊 comparison_results.json         # 详细对比结果
├── 📋 comparison_training_log.txt     # 完整训练日志
│
├── 📁 model_version_a_api/            # Version A模型
│   └── model.pt                       # API版本权重
│
├── 📁 model_version_b_scratch/        # Version B模型（从零实现）
│   └── model.pt                       # 从零版本权重
│
└── 📁 transformer_imdb_sentiment.py    # 单版本实现（之前的）
```