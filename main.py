import data_kdag
from model import LinearClassifier

X, y = data_kdag.get_data()

lc = LinearClassifier()

lc.train(X,y,200)

lc.evaluate(X,y)

lc.visualize(X,y,True)