from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
df.head()

df.to_csv('../data/raw/AB_NYC_2019.csv', index=False)

df.shape

df.info()

print("Existen 48895 registros y observamos que hay 16 columnas de entre las cuales vamos a usar para poder predecir el precio de cada apartamento.")
print("Podemos observar que las columnas last_review y reviews_per_month contienen mas de 1000 valores nulos.")
print("Los datos cuentan con 10 caracteristicas numericas y 6 caracteristicas categoricas")

print(df.drop("id", axis = 1).duplicated().sum())
print("Observamos que no hay registros duplicados")

df.drop(["id", "name", "host_id", "host_name", "last_review"], axis = 1, inplace = True)
df.head()

## Paso 3: Análisis de variables univariante

fig, axis = plt.subplots(2, 2, figsize = (10, 5))

sns.histplot(ax = axis[0, 0], data = df, x = "neighbourhood_group")
sns.histplot(ax = axis[0, 1], data = df, x = "neighbourhood").set(ylabel = None)
sns.histplot(ax = axis[1, 0], data = df, x = "room_type").set(ylabel = None)

plt.tight_layout()
plt.show()

df["neighbourhood"].value_counts()

df["room_type"].value_counts()

print("Observamos que hay 5 grupos de vecindarios pero los que tienen mas habitaciones son Brooklyn y Manhattan")

print("Observamos que los vecindarios con mayor numero de departamentos son: " \
"1. Williamsburg con 3920 departamentos. " \
"2. Bedford-Stuyvesant con 3714 departamentos. " \
"3. Harlem con 2658 departamentos.")

print("Observamos que hay 3 tipos de cuartos:" \
"1. Casa/apartamento entero. " \
"2. Apartamento privado. " \
"3. Apartamento compartido. " \
"Siendo este ultimo con menor cantidad de demanda.")

df.describe()

plt.figure(figsize=(10, 6))
plt.scatter(df['longitude'], df['latitude'], alpha=0.05, c=df['price'], cmap='viridis')
plt.colorbar(label='Price')
plt.title('Distribución geográfica de los alojamientos')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

Q1_price = df['price'].quantile(0.25)
Q3_price = df['price'].quantile(0.75)
IQR_price = Q3_price - Q1_price
Inferior = Q1_price - 1.5 * IQR_price
Superior = Q3_price + 1.5 * IQR_price

Q1_reviews = df['number_of_reviews'].quantile(0.25)
Q3_reviews = df['number_of_reviews'].quantile(0.75)
IQR_reviews = Q3_reviews - Q1_reviews
Inf = Q1_reviews - 1.5 * IQR_reviews
Sup = Q3_reviews + 1.5 * IQR_reviews

df_limpio = df[(df['price'] >= Inferior) & (df['price'] <= Superior)]
df_limpio = df_limpio[(df_limpio['number_of_reviews'] >= Inf) & (df_limpio['number_of_reviews'] <= Sup)]

plt.figure(figsize=(18, 9))

# Histograma de 'price' (eliminando outliers)
plt.subplot(2, 3, 1)
sns.histplot(df_limpio['price'], kde=True, bins=50, color='skyblue')
plt.title('Distribución de Precios')
plt.xlabel('Price')
plt.ylabel('Frequency')

# Histograma de 'minimum_nights'
plt.subplot(2, 3, 2)
sns.histplot(df['minimum_nights'], kde=False, bins=50, color='skyblue')
plt.title('Distribución de Noches Mínimas')
plt.xlabel('Minimum Nights')
plt.ylabel('Frequency')

# Histograma de 'number_of_reviews' (eliminando outliers)
plt.subplot(2, 3, 3)
sns.histplot(df['number_of_reviews'], kde=True, bins=50, color='orange')
plt.title('Distribución de Número de Reseñas')
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')

# Histograma de 'reviews_per_month'
plt.subplot(2, 3, 4)
sns.histplot(df['reviews_per_month'], kde=True, bins=50, color='green')
plt.title('Distribución de Reseñas por Mes')
plt.xlabel('Reviews per Month')
plt.ylabel('Frequency')

# Histograma de 'calculated_host_listings_count'
plt.subplot(2, 3, 5)
sns.histplot(df['calculated_host_listings_count'], kde=False, bins=50, color='purple')
plt.title('Distribución de Propiedades por Anfitrión')
plt.xlabel('Listings per Host')
plt.ylabel('Frequency')

# Histograma de 'availability_365'
plt.subplot(2, 3, 6)
sns.histplot(df['availability_365'], kde=True, bins=50, color='red')
plt.title('Distribución de Disponibilidad Anual')
plt.xlabel('Availability (Days per Year)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

## Paso 4: Análisis de variables multivariante

fig, axis = plt.subplots(4, 2, figsize = (10, 16))

sns.regplot(ax = axis[0, 0], data = df, x = "minimum_nights", y = "price")
sns.heatmap(df[["price", "minimum_nights"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)

sns.regplot(ax = axis[0, 1], data = df, x = "number_of_reviews", y = "price").set(ylabel = None)
sns.heatmap(df[["price", "number_of_reviews"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

sns.regplot(ax = axis[2, 0], data = df, x = "calculated_host_listings_count", y = "price").set(ylabel = None)
sns.heatmap(df[["price", "calculated_host_listings_count"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0]).set(ylabel = None)
fig.delaxes(axis[2, 1])
fig.delaxes(axis[3, 1])

plt.tight_layout()
plt.show()

print("No existe relacion entre el precio y las variablesnumericas anteriores")

fig, axis = plt.subplots(figsize = (5, 4))
sns.countplot(data = df, x = "room_type", hue = "neighbourhood_group")
plt.show()

print("Observamos que Brooklyn es el vecindario con mayor numero de habitaciones privadas para alquilar." \
"Manhattan es el vecindario con mayor numero de casas/apartamentos enteros para alquilar.")

df["room_type"] = pd.factorize(df["room_type"])[0]
df["neighbourhood_group"] = pd.factorize(df["neighbourhood_group"])[0]
df["neighbourhood"] = pd.factorize(df["neighbourhood"])[0]

fig, axes = plt.subplots(figsize=(15, 15))

sns.heatmap(df[["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights",	
                        "number_of_reviews", "calculated_host_listings_count", "availability_365"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()
plt.show()

## Paso 5: Ingeniería de características

df.describe()

fig, axis = plt.subplots(3, 3, figsize = (15, 10))

sns.boxplot(ax = axis[0, 0], data = df, y = "neighbourhood_group")
sns.boxplot(ax = axis[0, 1], data = df, y = "neighbourhood")
sns.boxplot(ax = axis[0, 2], data = df, y = "calculated_host_listings_count")
sns.boxplot(ax = axis[1, 0], data = df, y = "availability_365")
sns.boxplot(ax = axis[1, 1], data = df, y = "room_type")
sns.boxplot(ax = axis[1, 2], data = df, y = "price")
sns.boxplot(ax = axis[2, 0], data = df, y = "minimum_nights")
sns.boxplot(ax = axis[2, 1], data = df, y = "number_of_reviews")
sns.boxplot(ax = axis[2, 2], data = df, y = "reviews_per_month")

plt.tight_layout()
plt.show()

print("Se obserba que hay muchos valores que tienen outliers, entre los cuales tenemos el numero de reseñas, reseñas mensuales, minimo de noches, distribucion de propiedades por anfitrion y precio.")

price_stats = df["price"].describe()
price_stats

price_iqr = price_stats["75%"] - price_stats["25%"]
upper_limit = price_stats["75%"] + 1.5 * price_iqr
lower_limit = price_stats["25%"] - 1.5 * price_iqr

print(f"Los limites superior e inferior son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuantilico de {round(price_iqr, 2)}")

df = df[df["price"] > 0]

count_0 = df[df["price"] == 0].shape[0]
count_1 = df[df["price"] == 1].shape[0]

print("Contador 0: ", count_0)
print("Contador 1: ", count_1)

nights_stats = df["minimum_nights"].describe()
nights_stats

nights_iqr = nights_stats["75%"] - nights_stats["25%"]

limite_superior = nights_stats["75%"] + 1.5 * nights_iqr
limite_inferior = nights_stats["25%"] - 1.5 * nights_iqr

print(f"El limite superior e inferior para los outliers son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuantilico de {round(nights_iqr, 2)}")

df = df[df["minimum_nights"] <= 15]

count_0 = df[df["minimum_nights"] == 0].shape[0]
count_1 = df[df["minimum_nights"] == 1].shape[0]
count_2 = df[df["minimum_nights"] == 2].shape[0]
count_3 = df[df["minimum_nights"] == 3].shape[0]
count_4 = df[df["minimum_nights"] == 4].shape[0]


print("Contador de 0: ", count_0)
print("Contador de 1: ", count_1)
print("Contador de 2: ", count_2)
print("Contador de 3: ", count_3)
print("Contador de 4: ", count_4)

review_stats = df["number_of_reviews"].describe()
review_stats

review_iqr = review_stats["75%"] - review_stats["25%"]

limite_superior = review_stats["75%"] + 1.5 * review_iqr
limite_inferior = review_stats["25%"] - 1.5 * review_iqr

print(f"Los limites superior e inferior para los outliers son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuantilico de {round(review_iqr, 2)}")

hostlist_stats = df["calculated_host_listings_count"].describe()
hostlist_stats

hostlist_iqr = hostlist_stats["75%"] - hostlist_stats["25%"]

limite_superior = hostlist_stats["75%"] + 1.5 * hostlist_iqr
limite_inferior = hostlist_stats["25%"] - 1.5 * hostlist_iqr

print(f"Los limites superior e inferior para los outliers son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuantilico de {round(hostlist_iqr, 2)}")

count_0 = sum(1 for x in df["calculated_host_listings_count"] if x in range(0, 5))
count_1 = df[df["calculated_host_listings_count"] == 1].shape[0]
count_2 = df[df["calculated_host_listings_count"] == 2].shape[0]

print("Contador de 0: ", count_0)
print("Contador de 1: ", count_1)
print("Contador de 2: ", count_2)

df = df[df["calculated_host_listings_count"] > 4]

df.isnull().sum().sort_values(ascending = False)

from sklearn.preprocessing import MinMaxScaler

num_variables = ["number_of_reviews", "minimum_nights", "calculated_host_listings_count", 
                 "availability_365", "neighbourhood_group", "room_type"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(df[num_variables])
df_scal = pd.DataFrame(scal_features, index = df.index, columns = num_variables)
df_scal["price"] = df["price"]
df_scal.head()

## Paso 6: Selección de características

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

X = df_scal.drop("price", axis = 1)
y = df_scal["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


selection_model = SelectKBest(chi2, k = 4)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

X_train_sel.head()

X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)
X_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index = False)