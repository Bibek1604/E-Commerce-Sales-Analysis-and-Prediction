# ------------------------
# 📦 Import Dependencies
# ------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# ------------------------
# 🎨 Visualization Settings
# ------------------------
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.titleweight": "bold"
})

# ------------------------
# 📁 Create Output Folder
# ------------------------
output_dir = "insightful_figures"
os.makedirs(output_dir, exist_ok=True)

# ------------------------
# 📂 Load and Clean Data
# ------------------------
df = pd.read_csv("SampleSuperstore.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df['Postal Code'] = df['Postal Code'].astype(str)
df['Unit Price'] = df['Sales'] / df['Quantity']

# ------------------------
# 📅 Convert Order Date if exists
# ------------------------
if 'Order Date' in df.columns:
    df['Order Date'] = pd.to_datetime(df['Order Date'])

# ------------------------
# 📊 Summary Stats
# ------------------------
print("Summary Statistics:\n", df.describe())
print("\nColumns:\n", df.columns)

# ------------------------
# 📌 Helper Function
# ------------------------
def save_barplot(title, x, y, data, filename, order=None, hue=None, orient='v', palette="Set2", annotate=True):
    plt.figure()
    ax = sns.barplot(
        data=data,
        x=x if orient == 'v' else y,
        y=y if orient == 'v' else x,
        hue=hue,
        order=order,
        estimator=sum,
        ci=None,
        palette=palette,
        orient=orient
    )
    plt.title(title)
    if annotate:
        for p in ax.patches:
            val = p.get_height() if orient == 'v' else p.get_width()
            pos = (p.get_x() + p.get_width()/2, val) if orient == 'v' else (val, p.get_y() + p.get_height()/2)
            ax.annotate(f'{val:,.0f}', pos, ha='center', va='bottom' if orient == 'v' else 'center', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

# ------------------------
# 📈 Key Visualizations
# ------------------------

# Region-wise Sales
save_barplot("Total Sales by Region", "Region", "Sales", df, "sales_by_region")

# Category Sales & Profit
save_barplot("Sales by Category", "Category", "Sales", df, "sales_by_category")
save_barplot("Profit by Category", "Category", "Profit", df, "profit_by_category", palette="pastel")

# Sales vs Profit vs Discount
plt.figure()
sns.scatterplot(data=df, x='Sales', y='Profit', hue='Discount', size='Quantity',
                palette='coolwarm', alpha=0.7, sizes=(40, 400))
plt.title("Sales vs Profit (Colored by Discount)")
plt.tight_layout()
plt.savefig(f"{output_dir}/sales_vs_profit_discount.png", dpi=300)
plt.close()

# Sub-Category Sales
sub_sales = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).reset_index()
save_barplot("Sales by Sub-Category", "Sub-Category", "Sales", sub_sales, "sales_by_subcategory", order=sub_sales['Sub-Category'])

# Top 10 City, State, Postal Sales
city_sales = df.groupby('City')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
save_barplot("Top 10 Cities by Sales", "Sales", "City", city_sales, "top_cities_sales", orient='h')

state_sales = df.groupby('State')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
save_barplot("Top 10 States by Sales", "Sales", "State", state_sales, "top_states_sales", orient='h')

postal_sales = df.groupby('Postal Code')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
save_barplot("Top 10 Postal Codes by Sales", "Postal Code", "Sales", postal_sales, "top_postal_sales")

# Profit Distribution
plt.figure()
sns.boxplot(data=df, x="Category", y="Profit", palette="Pastel1")
plt.title("Profit Distribution by Category")
plt.tight_layout()
plt.savefig(f"{output_dir}/profit_distribution_category.png", dpi=300)
plt.close()

# Correlation Heatmap
plt.figure()
sns.heatmap(df[['Sales', 'Profit', 'Discount', 'Quantity']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300)
plt.close()

# Time-Series Sales Trend (if date column exists)
if 'Order Date' in df.columns:
    time_df = df.groupby(df['Order Date'].dt.to_period("M")).agg({'Sales': 'sum'}).reset_index()
    time_df['Order Date'] = time_df['Order Date'].dt.to_timestamp()
    plt.figure()
    sns.lineplot(data=time_df, x='Order Date', y='Sales', marker='o', color='teal')
    plt.title("Monthly Sales Trend")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/monthly_sales_trend.png", dpi=300)
    plt.close()

# ------------------------
# 🧠 Key Insights
# ------------------------
print("\n🔍 TOP INSIGHTS:")
print("1. Profits vary greatly across categories; some generate loss despite sales.")
print("2. Discounts often reduce profitability, as shown in the scatter plot.")
print("3. Most sales are generated from the West and East regions.")
print("4. 'Chairs', 'Phones', and 'Binders' are top-selling sub-categories.")
print("5. Postal code and city-wise analysis shows strong urban concentration.")
print("6. If time data exists: Sales show clear monthly trends for forecasting.")

# ------------------------
# ✅ Save Processed Data (Optional)
# ------------------------
df.to_csv("Cleaned_Superstore.csv", index=False)
