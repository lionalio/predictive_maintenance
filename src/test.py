from libs import *

with open('../data/processed/X_test.pkl', 'rb') as f3:
    X_test = pkl.load(f3)
with open('../data/processed/y_test.pkl', 'rb') as f4:
    y_test = pkl.load(f4)

model = keras.models.load_model('../models/transformer.keras')
print(X_test.shape)
print(y_test.shape)
print(y_test[0].value_counts())
print(model.evaluate(X_test, y_test, verbose=1))
preds = np.argmax(model.predict(X_test), axis=1)
print(np.unique(preds))