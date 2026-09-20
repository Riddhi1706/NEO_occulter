import pandas as pd # type: ignore

files = ["atira.csv", "aten.csv", "apollo.csv", "amor.csv"]

for file in files:
    df = pd.read_csv(file)

    print(f"\n{file}")
    print(f"Total objects: {len(df)}")
    print(df["pha"].value_counts(dropna=False))