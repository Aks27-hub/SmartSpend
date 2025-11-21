"""
choose_platform.py

Final version:
- PLATFORM DATA IS HARDCODED
- User inputs ONLY weights and optional constraints
- MILP selects the best platform

Run: python choose_platform.py
"""

import pandas as pd
import numpy as np
import pulp


# ---------------------------------------------------------
#  HARDCODED PLATFORM DATA (You can modify these values)
# ---------------------------------------------------------
def get_hardcoded_data():
    data = [
        {"platform":"Amazon",     "price":2599, "delivery":40,  "discount":180, "delivery_days":2, "rating":4.6, "return_score":0.92},
        {"platform":"Flipkart",   "price":2350, "delivery":75,  "discount":120, "delivery_days":3, "rating":4.1, "return_score":0.85},
        {"platform":"Meesho",     "price":1990, "delivery":150, "discount":60,  "delivery_days":6, "rating":3.6, "return_score":0.70},
        {"platform":"Croma", "price":2890, "delivery":0,   "discount":0,   "delivery_days":1, "rating":4.3, "return_score":0.88},
        {"platform":"JioMart",    "price":2450, "delivery":120, "discount":250, "delivery_days":4, "rating":4.0, "return_score":0.80},
        {"platform":"Myntra",    "price":3100, "delivery":90,  "discount":350, "delivery_days":5, "rating":4.7, "return_score":0.96},
        {"platform":"Blinkit",    "price":2780, "delivery":60,  "discount":50,  "delivery_days":3, "rating":3.8, "return_score":0.75}
    ]
    return pd.DataFrame(data)


# ----------------------------
# NORMALIZATION SUPPORT
# ----------------------------
def minmax(series):
    mn = series.min()
    mx = series.max()
    if mx == mn:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


def prepare_dataframe(df):
    df2 = df.copy().reset_index(drop=True)
    df2["price_n"] = minmax(df2["price"])
    df2["delivery_n"] = minmax(df2["delivery"])
    df2["discount_n"] = minmax(df2["discount"])
    df2["time_n"] = minmax(df2["delivery_days"])
    df2["rating_n"] = minmax(df2["rating"])
    df2["return_n"] = minmax(df2["return_score"])
    return df2


# ----------------------------
# COST COEFFICIENTS
# ----------------------------
def compute_cost_coeffs(df, w):
    df["cost_coef"] = (
        w["w_price"]   * df["price_n"] +
        w["w_delivery"]* df["delivery_n"] +
        w["w_time"]    * df["time_n"] -
        w["w_discount"]* df["discount_n"] -
        w["w_rating"]  * df["rating_n"] -
        w["w_return"]  * df["return_n"]
    )

    mn = df["cost_coef"].min()
    if mn < 0:
        df["cost_coef"] = df["cost_coef"] - mn

    return df


# ----------------------------
# MILP SOLVER
# ----------------------------
def solve_milp(df, max_delivery=None, min_rating=None):
    prob = pulp.LpProblem("PlatformSelectionMILP", pulp.LpMinimize)
    ids = df.index
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in ids}

    # Objective
    prob += pulp.lpSum(df.loc[i, "cost_coef"] * x[i] for i in ids)

    # Choose exactly one platform
    prob += pulp.lpSum(x[i] for i in ids) == 1

    # Hard constraints
    if max_delivery is not None:
        for i in ids:
            if df.loc[i, "delivery_days"] > max_delivery:
                prob += x[i] == 0

    if min_rating is not None:
        for i in ids:
            if df.loc[i, "rating"] < min_rating:
                prob += x[i] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    for i in ids:
        if pulp.value(x[i]) == 1:
            return df.loc[i].to_dict()

    return None


# ----------------------------
# MAIN PROGRAM
# ----------------------------
def main():
    print("=== PLATFORM OPTIMIZATION USING MILP ===")

    # Load hardcoded dataset
    df = get_hardcoded_data()
    print("\nHardcoded platform data:")
    print(df)

    # User weight inputs
    print("\nEnter your preference weights:")
    w_price    = float(input("Weight for price: "))
    w_delivery = float(input("Weight for delivery charge: "))
    w_discount = float(input("Weight for discount: "))
    w_time     = float(input("Weight for delivery time: "))
    w_rating   = float(input("Weight for rating: "))
    w_return   = float(input("Weight for return policy: "))

    w_raw = np.array([w_price, w_delivery, w_discount, w_time, w_rating, w_return])

    if w_raw.sum() == 0:
        print("Weights cannot be all zero. Using equal weights.")
        w_raw = np.ones(6)

    w_norm = w_raw / w_raw.sum()

    weights = {
        "w_price": w_norm[0],
        "w_delivery": w_norm[1],
        "w_discount": w_norm[2],
        "w_time": w_norm[3],
        "w_rating": w_norm[4],
        "w_return": w_norm[5]
    }

    print("\nNormalized weights used:", weights)

    # Optional constraints
    print("\nOptional constraints (press Enter to skip):")
    max_del = input("Max delivery days: ")
    min_rat = input("Min rating: ")

    max_delivery = float(max_del) if max_del.strip() else None
    min_rating = float(min_rat) if min_rat.strip() else None

    # Prepare data & compute cost
    df_norm = prepare_dataframe(df)
    df_coeff = compute_cost_coeffs(df_norm, weights)

    print("\nPlatform table with cost coefficients:")
    print(df_coeff[["platform", "cost_coef", "price", "delivery", "discount",
                    "delivery_days", "rating", "return_score"]])

    # Solve MILP
    print("\nSolving optimization...")
    chosen = solve_milp(df_coeff, max_delivery, min_rating)

    if chosen:
        print("\n=== BEST PLATFORM SELECTED ===")
        print("Platform:", chosen["platform"])
        print("\nDetails:")
        for k, v in chosen.items():
            print(f"{k}: {v}")
    else:
        print("\nNo platform satisfies the constraints.")


if __name__ == "__main__":
    main()
