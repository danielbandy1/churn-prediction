#!/usr/bin/env python3
"""Fetch the IBM Telco Customer Churn dataset; fall back to synthetic generation."""

import pathlib, urllib.request

MIRRORS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/telecom_churn.csv",
]
DEST = pathlib.Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already present: {DEST}  ({DEST.stat().st_size/1e3:.0f} KB)")
        return
    for url in MIRRORS:
        try:
            print(f"Trying {url} ...")
            urllib.request.urlretrieve(url, DEST)
            if DEST.stat().st_size > 10_000:
                print(f"Downloaded — {DEST.stat().st_size/1e3:.0f} KB")
                return
            DEST.unlink()
        except Exception as e:
            print(f"  failed: {e}")
    print("All mirrors failed — generating synthetic dataset...")
    import generate_data
    generate_data.main()


if __name__ == "__main__":
    main()
