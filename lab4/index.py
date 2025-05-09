import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_blobs, load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print("Wersja TensorFlow:", tf.__version__)
print("Wersja NumPy:", np.__version__)

# ---------------------------------------------------------------------------
# Zadanie 1: Optymalizacja funkcji f(x) = e^x - 2x metodą spadku gradientu
# ---------------------------------------------------------------------------
print("\n--- Zadanie 1: Optymalizacja Gradientowa ---")

def f(x):
  return np.exp(x) - 2 * x

def grad_f(x):
  return np.exp(x) - 2

def gradient_descent(grad_func, start_point, learning_rate, iterations):
  x = start_point
  history = [x]
  for _ in range(iterations):
    grad = grad_func(x)
    x = x - learning_rate * grad
    history.append(x)
  return x, history

initial_x = 3.0          
learning_rate_gd = 0.1   
iterations_gd = 50       

optimal_x, history_gd = gradient_descent(grad_f, initial_x, learning_rate_gd, iterations_gd)

analytical_min = np.log(2)
print(f"Punkt startowy: x = {initial_x}")
print(f"Znalezione minimum (numerycznie): x ≈ {optimal_x:.6f}")
print(f"Analityczne minimum: x = ln(2) ≈ {analytical_min:.6f}")
print(f"Wartość funkcji w znalezionym minimum: f(x) ≈ {f(optimal_x):.6f}")
print(f"Wartość funkcji w analitycznym minimum: f(ln(2)) ≈ {f(analytical_min):.6f}")


x_vals = np.linspace(optimal_x - 2, initial_x + 1, 400)
y_vals = f(x_vals)
history_vals = f(np.array(history_gd))

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=r'$f(x) = e^x - 2x$', zorder=1)
plt.scatter(history_gd, history_vals, color='red', label='Kroki optymalizacji', zorder=2, s=20)

for i in range(len(history_gd) - 1):
    plt.arrow(history_gd[i], history_vals[i],
              history_gd[i+1] - history_gd[i], history_vals[i+1] - history_vals[i],
              color='red', alpha=0.5, head_width=0.05, head_length=0.1, length_includes_head=True)

plt.scatter(optimal_x, f(optimal_x), color='green', s=100, zorder=3, label=f'Minimum numeryczne (x={optimal_x:.3f})')
plt.scatter(analytical_min, f(analytical_min), color='blue', marker='*', s=150, zorder=3, label=f'Minimum analityczne (x={analytical_min:.3f})')
plt.title('Optymalizacja funkcji $f(x) = e^x - 2x$ metodą spadku gradientu')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.ylim(min(y_vals) - 0.5, max(f(initial_x), max(y_vals)) + 0.5)
plt.show()


# ---------------------------------------------------------------------------
# Zadanie 2: Sieć neuronowa do klasyfikacji trzech klas (sztuczne dane)
# ---------------------------------------------------------------------------
print("\n--- Zadanie 2: Prosta Sieć Neuronowa do Klasyfikacji Wieloklasowej ---")


n_samples = 1000
n_features = 2
n_classes_nn = 3
centers = [(-2, -2), (0, 2), (2, -2)] 

X_nn, y_nn = make_blobs(n_samples=n_samples,
                       centers=centers,
                       n_features=n_features,
                       cluster_std=0.8, 
                       random_state=42)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_nn[:, 0], X_nn[:, 1], c=y_nn, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.7)
plt.title('Sztuczne dane dla klasyfikacji 3 klas')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')
plt.legend(handles=scatter.legend_elements()[0], labels=[f'Klasa {i}' for i in range(n_classes_nn)])
plt.grid(True)
plt.show()

y_nn_one_hot = to_categorical(y_nn, num_classes=n_classes_nn)

X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X_nn, y_nn_one_hot, test_size=0.2, random_state=42, stratify=y_nn_one_hot
)

print(f"Rozmiar zbioru treningowego: {X_train_nn.shape}")
print(f"Rozmiar zbioru testowego: {X_test_nn.shape}")

model_nn = keras.Sequential(
    [
        layers.Input(shape=(n_features,), name="warstwa_wejsciowa"),
        layers.Dense(16, activation="relu", name="warstwa_ukryta_1"), 
        layers.Dense(n_classes_nn, activation="softmax", name="warstwa_wyjsciowa") 
    ],
    name="prosta_siec_klasyfikacyjna"
)

model_nn.compile(
    loss="categorical_crossentropy", 
    optimizer="adam",                
    metrics=["accuracy"]             
)

model_nn.summary()

print("\nRozpoczęcie treningu modelu NN...")
epochs_nn = 30
batch_size_nn = 32

history_nn = model_nn.fit(
    X_train_nn,
    y_train_nn,
    epochs=epochs_nn,
    batch_size=batch_size_nn,
    validation_split=0.2, 
    verbose=1 
)
print("Trening zakończony.")

print("\nEwaluacja modelu NN na zbiorze testowym:")
loss_nn, accuracy_nn = model_nn.evaluate(X_test_nn, y_test_nn, verbose=0)
print(f"Strata (Loss) na zbiorze testowym: {loss_nn:.4f}")
print(f"Dokładność (Accuracy) na zbiorze testowym: {accuracy_nn:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_nn.history['accuracy'], label='Dokładność treningowa')
plt.plot(history_nn.history['val_accuracy'], label='Dokładność walidacyjna')
plt.title('Dokładność modelu NN podczas treningu')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_nn.history['loss'], label='Strata treningowa')
plt.plot(history_nn.history['val_loss'], label='Strata walidacyjna')
plt.title('Strata modelu NN podczas treningu')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nPrzykładowe prognozy:")
predictions_nn = model_nn.predict(X_test_nn[:5])
predicted_classes_nn = np.argmax(predictions_nn, axis=1)
true_classes_nn = np.argmax(y_test_nn[:5], axis=1)

for i in range(5):
  print(f"  Próbka {i+1}: Prawdziwa klasa = {true_classes_nn[i]}, Prognozowana klasa = {predicted_classes_nn[i]}, Prawdopodobieństwa = {predictions_nn[i].round(3)}")


# ---------------------------------------------------------------------------
# Zadanie 3: Sieć konwolucyjna (CNN) na zbiorze sklearn.datasets.load_digits
# ---------------------------------------------------------------------------
print("\n--- Zadanie 3: Sieć Konwolucyjna (CNN) dla zbioru Digits ---")

digits = load_digits()
X_cnn = digits.data
y_cnn = digits.target
n_samples_cnn = len(X_cnn)
n_classes_cnn = len(np.unique(y_cnn)) 

img_rows, img_cols = 8, 8
n_features_cnn = X_cnn.shape[1] 

print(f"Liczba próbek w zbiorze Digits: {n_samples_cnn}")
print(f"Liczba cech (pikseli): {n_features_cnn}")
print(f"Liczba klas: {n_classes_cnn}")
print(f"Wymiary obrazków: {img_rows}x{img_cols}")

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f'Etykieta: {digits.target[i]}')
    plt.axis('off')
plt.suptitle('Przykładowe obrazki ze zbioru Digits')
plt.show()


X_cnn_reshaped = X_cnn.reshape(n_samples_cnn, img_rows, img_cols, 1)
input_shape_cnn = (img_rows, img_cols, 1)

X_cnn_normalized = X_cnn_reshaped.astype('float32') / 16.0

y_cnn_one_hot = to_categorical(y_cnn, num_classes=n_classes_cnn)

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_cnn_normalized, y_cnn_one_hot, test_size=0.25, random_state=42, stratify=y_cnn
)

print(f"Rozmiar zbioru treningowego CNN: {X_train_cnn.shape}")
print(f"Rozmiar zbioru testowego CNN: {X_test_cnn.shape}")

model_cnn = keras.Sequential(
    [
        layers.Input(shape=input_shape_cnn, name="wejscie_cnn"),

        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'),
        
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        
        layers.Dropout(0.5),
        
        layers.Dense(128, activation="relu"),
        
        layers.Dropout(0.3),
        
        layers.Dense(n_classes_cnn, activation="softmax", name="wyjscie_cnn")
    ],
    name="prosta_siec_cnn_digits"
)

model_cnn.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model_cnn.summary()

print("\nRozpoczęcie treningu modelu CNN...")
epochs_cnn = 25
batch_size_cnn = 64

history_cnn = model_cnn.fit(
    X_train_cnn,
    y_train_cnn,
    epochs=epochs_cnn,
    batch_size=batch_size_cnn,
    validation_split=0.15, 
    verbose=1
)
print("Trening CNN zakończony.")

print("\nEwaluacja modelu CNN na zbiorze testowym:")
loss_cnn, accuracy_cnn = model_cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print(f"Strata (Loss) CNN na zbiorze testowym: {loss_cnn:.4f}")
print(f"Dokładność (Accuracy) CNN na zbiorze testowym: {accuracy_cnn:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='Dokładność treningowa CNN')
plt.plot(history_cnn.history['val_accuracy'], label='Dokładność walidacyjna CNN')
plt.title('Dokładność modelu CNN podczas treningu')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'], label='Strata treningowa CNN')
plt.plot(history_cnn.history['val_loss'], label='Strata walidacyjna CNN')
plt.title('Strata modelu CNN podczas treningu')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nPrzykładowe prognozy CNN i wizualizacja:")
predictions_cnn = model_cnn.predict(X_test_cnn)
predicted_classes_cnn = np.argmax(predictions_cnn, axis=1)
true_classes_cnn_labels = np.argmax(y_test_cnn, axis=1) 

plt.figure(figsize=(12, 8))
indices = np.random.choice(range(len(X_test_cnn)), size=15, replace=False) 

for i, index in enumerate(indices):
    plt.subplot(3, 5, i + 1)
    plt.imshow(X_test_cnn[index].reshape(img_rows, img_cols), cmap=plt.cm.gray_r, interpolation='nearest')
    true_label = true_classes_cnn_labels[index]
    pred_label = predicted_classes_cnn[index]
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f'Prawda: {true_label}\nPred: {pred_label}', color=color)
    plt.axis('off')

plt.suptitle('Przykładowe wyniki predykcji CNN na zbiorze testowym Digits')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.show()

print("\nZakończono wszystkie zadania dla Wariantu 4.")
