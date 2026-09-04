import numpy as np

# 1. Data Load karna (Price=col2, Bed=col3, Bath=col4)
data = np.genfromtxt("week1/RealEstate-USA.csv", delimiter=',', 
                     skip_header=1, usecols=(2, 3, 4), 
                     filling_values=np.nan)

# Khali jagah ko 0 se badalna takay math operations chal sakein
data = np.nan_to_num(data, nan=0)
price = data[:, 0]
bed = data[:, 1]
bath = data[:, 2]

# --- STATISTICS OPERATIONS ---
print("Mean Price:", np.mean(price))
print("Median Price:", np.median(price))
print("Standard Deviation:", np.std(price))
print("25th Percentile:", np.percentile(price, 25))

# --- MATH OPERATIONS ---
# Square, Sqrt, Abs
print("Price Square:", np.square(price[:5])) # Sirf pehli 5 dikha rahe hain
print("Price Sqrt:", np.sqrt(price[:5]))
print("Price Abs:", np.abs(price[:5]))

# --- TRIGONOMETRIC OPERATIONS ---
# Hum price ko pi se divide karke trig functions apply kar rahe hain
pricePie = (price / np.pi) + 1
print("Sine values (first 5):", np.sin(pricePie[:5]))
print("Logarithm (Natural):", np.log(pricePie[:5]))

# --- 2D ARRAY OPERATIONS (Reshape & Slicing) ---
# Bed aur Bath ka 2D array
D2BedBath = np.array([bed, bath])

print("\n2D Array Shape:", D2BedBath.shape)
print("Total Elements:", D2BedBath.size)

# Reshape (2 x N -> 1 x 2N)
D2Reshaped = np.reshape(D2BedBath, (1, -1))
print("New Shape after Reshape:", D2Reshaped.shape)

# Slicing (Pehli 2 rows aur pehle 5 columns)
slice_data = D2BedBath[:, 0:5]
print("Slice of 2D Array:", slice_data)