# Data

Download the M5 Walmart retail forecasting files from the official competition source and place them in `data/raw/`:

- `sales_train_evaluation.csv`
- `sell_prices.csv`
- `calendar.csv`

These large CSV files are ignored by Git. Run `python model_training.py` to create the local prepared dataset in `data/processed/`.