import csv
import random
from datetime import datetime, timedelta

# Configuration
START_DATE = datetime(2023, 1, 1)
DAYS_TO_SIMULATE = 365
OUTPUT_FILE = 'restaurant_data.csv'

# Define Ingredients and their initial stock
# Format: 'Ingredient Name': {'unit': 'unit_name', 'initial_stock': quantity, 'restock_amount': quantity, 'restock_threshold': quantity}
INGREDIENTS = {
    'Pizza Dough': {'unit': 'ball', 'initial_stock': 100, 'restock_amount': 50, 'restock_threshold': 20},
    'Tomato Sauce': {'unit': 'liter', 'initial_stock': 50, 'restock_amount': 20, 'restock_threshold': 10},
    'Mozzarella Cheese': {'unit': 'kg', 'initial_stock': 30, 'restock_amount': 15, 'restock_threshold': 5},
    'Pepperoni': {'unit': 'kg', 'initial_stock': 20, 'restock_amount': 10, 'restock_threshold': 5},
    'Pasta': {'unit': 'kg', 'initial_stock': 50, 'restock_amount': 25, 'restock_threshold': 10},
    'Ground Beef': {'unit': 'kg', 'initial_stock': 40, 'restock_amount': 20, 'restock_threshold': 10},
    'Lettuce': {'unit': 'head', 'initial_stock': 50, 'restock_amount': 30, 'restock_threshold': 10},
    'Tomatoes': {'unit': 'kg', 'initial_stock': 30, 'restock_amount': 15, 'restock_threshold': 5},
    'Burger Bun': {'unit': 'piece', 'initial_stock': 100, 'restock_amount': 50, 'restock_threshold': 20},
    'Burger Patty': {'unit': 'piece', 'initial_stock': 80, 'restock_amount': 40, 'restock_threshold': 15},
    'Onion': {'unit': 'kg', 'initial_stock': 25, 'restock_amount': 10, 'restock_threshold': 5}
}

# Define Menu and Recipes
# Format: 'Dish Name': {'Ingredient Name': quantity_needed, ...}
MENU = {
    'Margherita Pizza': {
        'Pizza Dough': 1,
        'Tomato Sauce': 0.2, # liters
        'Mozzarella Cheese': 0.25 # kg
    },
    'Pepperoni Pizza': {
        'Pizza Dough': 1,
        'Tomato Sauce': 0.2,
        'Mozzarella Cheese': 0.2,
        'Pepperoni': 0.1
    },
    'Spaghetti Bolognese': {
        'Pasta': 0.2, # kg
        'Tomato Sauce': 0.15,
        'Ground Beef': 0.15,
        'Onion': 0.05
    },
    'Caesar Salad': {
        'Lettuce': 0.5, # head
        'Tomatoes': 0.2, # kg
        'Mozzarella Cheese': 0.05
    },
    'Cheeseburger': {
        'Burger Bun': 1,
        'Burger Patty': 1,
        'Mozzarella Cheese': 0.05,
        'Lettuce': 0.1,
        'Tomatoes': 0.05,
        'Onion': 0.02
    }
}

def generate_data():
    current_date = START_DATE
    current_stock = {ing: data['initial_stock'] for ing, data in INGREDIENTS.items()}
    
    # List to store the generated data rows
    data_rows = []
    
    # Simulate each day
    for day in range(DAYS_TO_SIMULATE):
        date_str = current_date.strftime('%Y-%m-%d')
        
        # 1. Restock if needed (Simplistic Logic: Restock at start of day)
        for ing, stock in current_stock.items():
            if stock < INGREDIENTS[ing]['restock_threshold']:
                restock_amt = INGREDIENTS[ing]['restock_amount']
                current_stock[ing] += restock_amt
                # Optionally log restock events, but for now we focus on order data
        
        # 2. Simulate Orders for the day
        # Random number of orders between 20 and 50
        num_orders = random.randint(20, 50)
        
        for _ in range(num_orders):
            # Pick a random dish
            dish_name = random.choice(list(MENU.keys()))
            recipe = MENU[dish_name]
            
            # Check if we have enough ingredients
            can_make = True
            for ing, qty_needed in recipe.items():
                if current_stock[ing] < qty_needed:
                    can_make = False
                    break
            
            if can_make:
                # Generate a unique Order ID (e.g., specific to day and sequence)
                order_id = f"ORD-{date_str.replace('-', '')}-{random.randint(1000, 9999)}"
                
                # Deduct stock and record data
                for ing, qty_needed in recipe.items():
                    current_stock[ing] -= qty_needed
                    # Ensure stock doesn't go below zero (floating point correction)
                    if current_stock[ing] < 0: current_stock[ing] = 0
                    
                    row = {
                        'Date': date_str,
                        'Order_ID': order_id,
                        'Dish_Name': dish_name,
                        'Ingredient_Name': ing,
                        'Quantity_Used': qty_needed,
                        'Unit': INGREDIENTS[ing]['unit'],
                        'Stock_Available': round(current_stock[ing], 2)
                    }
                    data_rows.append(row)
            else:
                # Stock out - Order lost (Optional: limit output to successful orders)
                pass
        
        # Move to next day
        current_date += timedelta(days=1)

    # Write to CSV
    fieldnames = ['Date', 'Order_ID', 'Dish_Name', 'Ingredient_Name', 'Quantity_Used', 'Unit', 'Stock_Available']
    
    with open(OUTPUT_FILE, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)
        
    print(f"Data generation complete. Saved {len(data_rows)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_data()
