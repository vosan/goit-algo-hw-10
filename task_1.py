"""
The company produces two drinks: Lemonade and Fruit Juice.
Goal: maximize the total number of produced units to resource limits.

Resources available:
- Water: 100 units
- Sugar: 50 units
- Lemon juice: 30 units
- Fruit purée: 40 units

Recipes (units of resource per unit of product):
- Lemonade: 2 Water, 1 Sugar, 1 Lemon juice
- Fruit Juice: 1 Water, 0 Sugar, 0 Lemon juice, 2 Fruit purée

Expected optimal solution:
- Lemonade = 30 units
- Fruit Juice = 20 units
- Total = 50 units
"""
import pulp


def build_and_solve() -> tuple[float, float, float]:
    """Build and solve the linear program using PuLP.

    Returns:
        A tuple (lemonade, fruit_juice, total) with optimal production quantities.
    """
    # Create a maximization problem.
    model = pulp.LpProblem("Drink_Production_Optimization", pulp.LpMaximize)

    # Decision variables: number of units to produce (non-negative, continuous by default).
    lemonade = pulp.LpVariable("Lemonade", lowBound=0)
    fruit_juice = pulp.LpVariable("FruitJuice", lowBound=0)

    # Objective: maximize total production.
    model += lemonade + fruit_juice, "Maximize_Total_Units"

    # Resource constraints.
    # Water: 2 per Lemonade, 1 per FruitJuice, total <= 100
    model += 2 * lemonade + 1 * fruit_juice <= 100, "Water_Limit"

    # Sugar: 1 per Lemonade, total <= 50
    model += lemonade <= 50, "Sugar_Limit"

    # Lemon juice: 1 per Lemonade, total <= 30
    model += lemonade <= 30, "Lemon_Juice_Limit"

    # Fruit puree: 2 per FruitJuice, total <= 40
    model += 2 * fruit_juice <= 40, "Fruit_Puree_Limit"

    # Solve the model using the default solver (CBC bundled with PuLP, if available).
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract solution values; fallback to 0.0 if variable has no value.
    lemonade_val = float(pulp.value(lemonade) or 0.0)
    fruit_juice_val = float(pulp.value(fruit_juice) or 0.0)
    total_val = lemonade_val + fruit_juice_val

    return lemonade_val, fruit_juice_val, total_val


def main() -> None:
    """Run the optimization and print results."""
    lemonade, fruit_juice, total = build_and_solve()

    # Print a concise report.
    print("Optimal production plan:")
    print(f"  Lemonade: {lemonade:.0f} units")
    print(f"  Fruit Juice: {fruit_juice:.0f} units")
    print(f"  Total: {total:.0f} units")


if __name__ == "__main__":
    main()
