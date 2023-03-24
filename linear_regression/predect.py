import pickle
import os

def estimate_price(mileage, theta):
    """Estimate the price of a car given its mileage and theta."""
    return theta[0] + theta[1] * mileage

def main():
    theta0 = 0
    theta1 = 0
    try:
        mile = float(input("Enter the mileage: "))
        if os.path.isfile("theta.p"):
            theta1, theta0, min_miles, max_miles, min_price, max_price = pickle.load(open("theta.p","rb"))
            mile_scaled = (mile - min_miles) / (max_miles - min_miles)
            price = estimate_price(mile_scaled, [theta0, theta1])
            price = price * (max_price - min_price) + min_price
        else:
            price = estimate_price(mile, [theta0, theta1])
        if price < 0:
            print("the predicted price is negative, the car is probably stolen or the range of the dataset is too small")
        else:
            print(f"Estimated price: {price}")
    except ValueError:
        print("Please enter a float number")
        return

if __name__ == "__main__":
    main()
