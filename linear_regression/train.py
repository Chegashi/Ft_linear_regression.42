
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import minmax_scaling
from tqdm import tqdm
import pickle

class train:
  def louad_data(self:object, path: str):
    try:
      df = pd.read_csv(path)
      self.miles =  df["km"].to_numpy()
      self.price =  df["price"].to_numpy()
      self.miles = self.miles.reshape(self.miles.shape[0], 1)
      self.price = self.price.reshape(self.price.shape[0], 1)
      self.scaled_price = minmax_scaling(self.price, columns=[0])
      self.scaled_miles = minmax_scaling(self.miles, columns=[0])
      plt.scatter(self.miles, self.price) 
      plt.xlabel('km')
      plt.ylabel('price')
      plt.title('brut data')
      plt.savefig('brut_data.png')

      plt.xlabel('km')
      plt.ylabel('price')
      plt.title('scaling data')
      plt.scatter(self.scaled_price, self.scaled_miles)
      plt.savefig('scaling_data.png')

    except Exception as e:
      print(e)
      return None

  def model(self, X, theta):
    return X.dot(theta)

  def cost_function(self, X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((self.model(X, theta) - y)**2)

  def grad(self, X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros((n_iterations, 3), dtype=float)
    
    x_ = np.linspace(min(self.scaled_miles), max(self.scaled_miles), 100)

    
    m = len(y)
    fig = plt.figure()
    for i in tqdm(range(n_iterations)):
      theta = theta - learning_rate * 1/m * X.T.dot(self.model(X, theta) - self.scaled_price)
      y_ = theta[0] * x_ + theta[1]
      plt.plot(x_, y_, c='b', alpha=0.002)
      cost_history[i] = [self.cost_function(X, y, theta) , theta[0][0], theta[1][0]]

    plt.plot(x_, y_, c='g')
    plt.scatter(self.scaled_miles, self.scaled_price, c='r', alpha = 1)
    plt.xlabel('km')
    plt.ylabel('price')
    plt.title('variation of h with a and b')
    plt.savefig('hypotesis_variation.png')

    return theta, cost_history

  def coef_determination(self, y, pred):
      u = ((y - pred)**2).sum()
      v = ((y - y.mean())**2).sum()
      return 1 - u/v

  def main_traing(self:object):
    theta0 = 0
    theta1 = 0
    n_iterations = 100000
    learning_rate = 0.001

    theta = np.array([theta1, theta0]).reshape(2, 1)
    self.X = np.hstack((self.scaled_miles, np.ones(self.scaled_miles.shape)))
    theta_final, cost_history = self.grad(self.X, self.scaled_price, theta, learning_rate, n_iterations)
    j_ab = np.array(cost_history)

    fig = plt.figure()
    a = cost_history[:,1]
    b = cost_history[:,2]
    h = cost_history[:,0]

    plt.plot(a, h)
    plt.xlabel('a | b - axis')
    plt.ylabel('h(a) - axis')
    plt.title('variation of h(a) with a and b')

    plt.plot(b, h)
    plt.legend(['h(a)', 'h(b)'])
    plt.savefig('hypotesis.png')

    j_A_B = np.array(cost_history)
    a = cost_history[:,1]
    b = cost_history[:,2]
    h = cost_history[:,0]

    fig = plt.figure()
    plt.plot(range(n_iterations), a)
    plt.ylabel('a - axis')
    plt.xlabel('N- iteration - axis')
    plt.savefig('a(N-iteration).png')
    plt.title('variation of a with N-iteration')

    fig = plt.figure()
    plt.plot(range(n_iterations), b)
    plt.ylabel('b - axis')
    plt.xlabel('N- iteration - axis')
    plt.savefig('b(N-iteration).png')
    plt.title('variation of b with N-iteration')

    fig = plt.figure()
    plt.plot(range(n_iterations), h)
    plt.ylabel('h - axis')
    plt.xlabel('N- iteration - axis')
    plt.savefig('h(N-iteration).png')
    plt.title('variation of h with N-iteration')

    predictions = self.model(self.X, theta_final)
    print(f"precision of the algorithme {self.coef_determination(self.scaled_price, predictions)}")

    fig = plt.figure()
    x_ = np.linspace(min(self.scaled_miles), max(self.scaled_miles), 100)
    y_ = theta_final[0] * x_ + theta_final[1]

    plt.plot(x_, y_, c='g', label='Prediction')
    plt.scatter(self.scaled_miles, self.scaled_price, c='r', label='Training Data')
    plt.xlabel('km')
    plt.ylabel('price')
    plt.savefig('prediction.png')
    return theta_final[0], theta[1]

def main():
  model = train()
  model.louad_data("data.csv")
  theta1, theta0 = model.main_traing()
  print("saving figures and model to visualize the model ...")
  min_miles = model.miles.min()
  max_miles = model.miles.max()
  min_price = model.price.min()
  max_price = model.price.max()
  pickle.dump([theta1, theta0, min_miles, max_miles, min_price, max_price], open("theta.p", "wb"))


if __name__ == "__main__":
  main()