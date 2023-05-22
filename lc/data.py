import pickle

d = "../data/GEOM/QM9/val_data_5k.pkl"
with open(d, "rb") as f:
    data = pickle.load(f)
print(len(data))
