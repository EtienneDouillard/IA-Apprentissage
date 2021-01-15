# Le jeu de données des images est téléchargeable directement à partir
# d'une fonction scikit-learn. On peut donc directement obtenir ce dataset via un appel de fonction :
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import neighbors  # utilisation d'un model déja existant dans sklearn c'est le model neighbors.
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1)

# Le dataset principal qui contient toutes les images
# mnist contient deux entrées principales
# print (mnist.data.shape)


# Data contient les images sous forme de tableaux de 28 x 28 = 784 couleurs de pixel en niveau de gris,
# Entre 0 et 16 (0 = blanc, 16 = noir).
# target Le vecteur d'annotations associé au dataset (nombre entre 0 et 9) Qui correspond à la valauer "lue" du chiffre
# print (mnist.target.shape)

# ffectuer un sampling et travailler sur seulement 5000 données :
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]

# séparer le jeu de données en training set et testing set. Avec la librairy train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)

#Le model de neighbors déja existant dans la librairie
knn = neighbors.KNeighborsClassifier(n_neighbors=3) #3 nombre de voisins
knn.fit(xtrain, ytrain)

#l'algorithme sauvegarde toutes les données en mémoire. C'est sa manière d'apprendre en quelque sorte.
error = 1 - knn.score(xtest, ytest) # knn.score renvoie  le pourcentage de prédiction véridique trouvée par le classifieur
print('Erreur: %f' % error)

#Pour trouver le k optimal, on va testle modèle pour tous les k de 2 à 15,
# On mesure l’erreur test et on affiche la performance en fonction de k
errors = []
for k in range(2,15):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(xtrain, ytrain).score(xtest, ytest)))
plt.plot(range(2,15), errors, 'o-')
plt.show()

#après le plot on remarque que le Knn le plus perfomant est avec k=4
# On récupère le classifieur le plus performant
knn = neighbors.KNeighborsClassifier(4)
knn.fit(xtrain, ytrain)

# On récupère les prédictions sur les données test
predicted = knn.predict(xtest)

# On redimensionne les données sous forme d'images
images = xtest.reshape((-1, 28, 28))

# On selectionne un echantillon de 12 images au hasard
select = np.random.randint(images.shape[0], size=12)

# On affiche les images avec la prédiction associée
fig,ax = plt.subplots(3,4)

for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: {}'.format( predicted[value]) )

plt.show()