import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("./../data/urls_features.csv")
    # Podział całego DataFrame: 70% train, 30% tymczasowy
    df_train, df_temp = train_test_split(
        df,
        test_size=0.30,
        stratify=df["type"],
        random_state=42,
        shuffle=True
    )

    # Podział tymczasowego zbioru na validation i test
    # 50% z 30% = 15% całego datasetu
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        stratify=df_temp["type"],
        random_state=42,
        shuffle=True
    )

    # Podsumowanie
    print(f"Train:      {len(df_train)} ({len(df_train) / len(df) * 100:.2f}%)")
    print(f"Validation: {len(df_val)} ({len(df_val) / len(df) * 100:.2f}%)")
    print(f"Test:       {len(df_test)} ({len(df_test) / len(df) * 100:.2f}%)")

    df_train.to_csv('./../data/train.csv',index=False, encoding="utf-8")
    df_val.to_csv('./../data/validation.csv',index=False, encoding="utf-8")
    df_test.to_csv('./../data/test.csv',index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
