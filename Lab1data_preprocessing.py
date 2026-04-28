import pandas as pd

pd.set_option("display.max_columns", None)   # show all columns
pd.set_option("display.width", 120)          # wider terminal output


# ── Step 1: Noise handling ─────────────────────────────────────────────────────
def noise_handle_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["city"] = cleaned["city"].astype("string").str.strip().str.title()
    cleaned["purchased"] = (
        cleaned["purchased"].astype("string").str.strip().str.lower()
    )
    return cleaned


# ── Step 2: Remove duplicates ──────────────────────────────────────────────────
def remove_duplication(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


# ── Step 3: Fill missing values ────────────────────────────────────────────────
def replace_missing_value_by_median(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in ["age", "income"]:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    cleaned["city"] = cleaned["city"].fillna(cleaned["city"].mode()[0])
    return cleaned


# ── Step 4: Drop remaining NaN ─────────────────────────────────────────────────
def remove_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


# ── Step 5: Outlier removal (IQR) ─────────────────────────────────────────────
def outlier(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in ["age", "income"]:
        q1 = cleaned[col].quantile(0.25)
        q3 = cleaned[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned = cleaned[(cleaned[col] >= lower) & (cleaned[col] <= upper)]
    return cleaned


# ── Step 6: Discretization ─────────────────────────────────────────────────────
def discretization(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["age_group"] = pd.cut(
        cleaned["age"],
        bins=[0, 21, 24, 100],
        labels=["Young", "Adult", "Senior"]
    )
    cleaned["income_group"] = pd.cut(
        cleaned["income"],
        bins=[0, 27000, 31000, 100000],
        labels=["Low", "Medium", "High"],
    )
    return cleaned


# ── Display helper ─────────────────────────────────────────────────────────────
def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def view(df: pd.DataFrame) -> None:
    section("FINAL PROCESSED DATA — All Rows & Columns")
    print(df.to_string(index=True))

    section("COLUMN-BY-COLUMN VALUES")
    for col in df.columns:
        print(f"\n  [{col}]")
        for i, val in enumerate(df[col].values):
            print(f"    Row {i}: {val}")

    section("STATISTICAL SUMMARY")
    print(df.describe(include="all").to_string())

    section("DATA TYPES & NON-NULL COUNTS")
    df.info()


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main() -> None:
    # Raw dataset (embedded — no file needed)
    data = pd.DataFrame({
        "name":      ["Asha", "Bikash", "Asha", "Chandra", "Dipesh", "Elina"],
        "age":       [21, None, 21, 24, 23, None],
        "income":    [25000, 32000, 25000, None, 28000, 31000],
        "city":      ["Kathmandu", "pokhara", "Kathmandu", "Lalitpur ", None, "pokhara"],
        "purchased": ["Yes", "No", "Yes", "No", "Yes", "No"],
    })

    section("RAW DATA (before any processing)")
    print(data.to_string(index=True))
    print(f"\nMissing values:\n{data.isnull().sum()}")

    # Pipeline
    processed = noise_handle_data(data)
    section("AFTER STEP 1 — Noise Handling (text cleaned)")
    print(processed[["name", "city", "purchased"]].to_string())

    processed = remove_duplication(processed)
    section("AFTER STEP 2 — Duplicates Removed")
    print(f"  Rows remaining: {len(processed)}")
    print(processed.to_string())

    processed = replace_missing_value_by_median(processed)
    section("AFTER STEP 3 — Missing Values Filled")
    print(processed.to_string())

    processed = remove_missing_data(processed)
    section("AFTER STEP 4 — Remaining NaN Dropped")
    print(processed.to_string())

    processed = outlier(processed)
    section("AFTER STEP 5 — Outliers Removed")
    print(processed.to_string())

    processed = discretization(processed)
    section("AFTER STEP 6 — Discretization")
    print(processed[["name", "age", "age_group", "income", "income_group"]].to_string())

    # Encoding & scaling
    processed["purchased"] = processed["purchased"].map({"yes": 1, "no": 0})
    processed["income_scaled"] = (
        (processed["income"] - processed["income"].min())
        / (processed["income"].max() - processed["income"].min())
    )

    # Final rich view
    view(processed)


if __name__ == "__main__":
    main()