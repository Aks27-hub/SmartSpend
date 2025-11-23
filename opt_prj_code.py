import pandas as pd
import numpy as np
import pulp

""" 
----------------------------
It is assumed for this project that a single product has to be purchased
----------------------------
"""

# get PLATFORM DATA
def platform_data():
    data = [
        {"platform":"Amazon",   "price":2750, "delivery_fee":120, "discount":150, "delivery_days":4, "rating":4.2, "return_score":0.83},
        {"platform":"Flipkart", "price":2400, "delivery_fee":60,  "discount":80,  "delivery_days":3, "rating":4.3, "return_score":0.86},
        {"platform":"Meesho",   "price":1850, "delivery_fee":140, "discount":50,  "delivery_days":6, "rating":3.8, "return_score":0.70},
        {"platform":"Croma",    "price":3100, "delivery_fee":70,   "discount":20,  "delivery_days":2, "rating":4.6, "return_score":0.89},
        {"platform":"JioMart",  "price":2250, "delivery_fee":130, "discount":200, "delivery_days":5, "rating":3.9, "return_score":0.77},
        {"platform":"Myntra",   "price":2950, "delivery_fee":90,  "discount":280, "delivery_days":5, "rating":4.4, "return_score":0.82},
        {"platform":"Blinkit",  "price":2600, "delivery_fee":30,  "discount":70,  "delivery_days":1, "rating":4.0, "return_score":0.79}
    ]
    return pd.DataFrame(data)


# min-max NORMALIZATION
def normalize(series):
    mini = series.min()
    maxi = series.max()
    if maxi == mini:
        return pd.Series(0.0, index=series.index)
    return (series - mini) / (maxi - mini)


# make a new normalized dataframe from old dataframe
def make_dataframe(df):
    df2 = df.copy().reset_index(drop=True)
    df2["price_n"] = normalize(df2["price"])
    df2["delivery_n"] = normalize(df2["delivery_fee"])
    df2["discount_n"] = normalize(df2["discount"])
    df2["time_n"] = normalize(df2["delivery_days"])
    df2["rating_n"] = normalize(df2["rating"])
    df2["return_n"] = normalize(df2["return_score"])
    return df2


# COST COEFFICIENTS
def compute_cost_coeff(df, w):
    df["cost_coeff"] = ( w["w_price"] * df["price_n"] + w["w_delivery"] * df["delivery_n"] +
                        w["w_time"] * df["time_n"] - w["w_discount"] * df["discount_n"] -
                        w["w_rating"] * df["rating_n"] - w["w_return"] * df["return_n"] )

    return df


# MILP SOLVER
def solve_MILP(df):
    p = pulp.LpProblem("PlatformSelectionMILP", pulp.LpMinimize)
    ids = df.index
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in ids}

    # Objective function
    p += pulp.lpSum(df.loc[i, "cost_coeff"] * x[i] for i in ids)

    # Choose exactly one platform
    p += pulp.lpSum(x[i] for i in ids) == 1

    p.solve(pulp.PULP_CBC_CMD(msg=False))

    for i in ids:
        if pulp.value(x[i]) == 1:
            return df.loc[i].to_dict()

    return None


def main():
    print("=== PLATFORM OPTIMIZATION USING MILP ===")

    df = platform_data()
    print("\nPlatform data:")
    print(df)

    print("\nEnter your preference weights (0-1):")
    w_price    = float(input("Weight for price: "))
    w_delivery = float(input("Weight for delivery charge: "))
    w_discount = float(input("Weight for discount: "))
    w_time     = float(input("Weight for delivery time: "))
    w_rating   = float(input("Weight for rating: "))
    w_return   = float(input("Weight for return policy: "))

    raw_weights = np.array([w_price, w_delivery, w_discount, w_time, w_rating, w_return])

    if raw_weights.sum() == 0:
        print("Weights cannot be all zero. Using equal weights (all 1).")
        raw_weights = np.ones(6)

    # normalized weights
    w_norm = raw_weights / raw_weights.sum()

    weights = {"w_price": w_norm[0], "w_delivery": w_norm[1], "w_discount": w_norm[2],
               "w_time": w_norm[3], "w_rating": w_norm[4], "w_return": w_norm[5]}

    print("\nNormalized weights:", weights)

    # Prepare data & compute cost
    df_norm = make_dataframe(df)
    df_coeff = compute_cost_coeff(df_norm, weights)

    print("\nPlatform data with cost coefficients:")
    print(df_coeff[["platform", "cost_coeff", "price", "delivery_fee", "discount",
                    "delivery_days", "rating", "return_score"]])

    # Solve MILP
    print("\nSolving optimization...")
    chosen = solve_MILP(df_coeff)

    if chosen:
        print("\n=== BEST PLATFORM SELECTED ===")
        print("Platform:", chosen["platform"])
        print("\nDetails:")
        for i, j in chosen.items():
            if (not i.endswith('n') and not i.startswith('cost')):
                print(f"{i}: {j}")
    else:
        print("\nNo platform satisfies the constraints.")

main()
