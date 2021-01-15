import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection #import du model de sélection
from sklearn import preprocessing
from sklearn import neighbors, metrics#Validation croisée


###Utiliser sklearn pour faire de la sélection de modele


data = pd.read_csv('winequality.csv', sep=";")
#print(data)

#point des données
X = data[data.columns[:-1]].values
#étiquette quality
y = data['quality'].values
#print(X)
#print(y)



#Séparer les bon vins par les vins médiocres par une méthode de classification

y_class = np.where(y<6,0,1)#si y<6 y=0 (vin pas terrible)  si y>6 y=1 (vin bon)
#séparation des données en training set et testing set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_class,test_size=0.3) # 30% des données dans le jeu de test

# standardiser les données d’entraînement et appliquer la même transformation aux données de test :
#StandarScaler()= Normaliser (Centrer et normer)  = (x - Moyenne) / Variance
std_scale = preprocessing.StandardScaler().fit(X_train) #fit(X_train) = apprendre sur le jeu d'entrainement
X_train_std = std_scale.transform(X_train)#transformation du jeu d'entrainement en appliquant le jeu de test
X_test_std = std_scale.transform(X_test)


#Affichage des données Normalisées sous forme d'histogrammes
fig = plt.figure(figsize=(16, 12))
for feat_idx in range(X_train_std.shape[1]):
    ax = fig.add_subplot(3,4, (feat_idx+1))#4diagrammes sur 3 lignes
    h = ax.hist(X_train_std[:, feat_idx], bins=50, color='steelblue', density = True, edgecolor='none')#gcréation des historammes
    ax.set_title(data.columns[feat_idx], fontsize=20)
plt.show()


#utiliser la méthode "GridSearchCV" pour faire une validation croisée afin de choisir le paramètre k d’un kNN sur le jeu d’entraînement
from sklearn import neighbors, metrics

# Fixer les valeurs des hyperparamètres à tester
param_grid = {'n_neighbors':[3, 5, 7, 9, 11, 13, 15]}

# Choisir un score à optimiser, ici l'accuracy (proportion de prédictions correctes)
score = 'accuracy' #proportion de point correctement classifié

# Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée
clf = model_selection.GridSearchCV(
    neighbors.KNeighborsClassifier(), # un classifieur kNN
    param_grid,     # hyperparamètres à tester
    cv=5,           # nombre de folds de validation croisée
    scoring=score   # score à optimiser
)

# Optimiser ce classifieur sur le jeu d'entraînement
clf.fit(X_train_std, y_train)

# Afficher le(s) hyperparamètre(s) optimaux
print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
print(clf.best_params_)

# Afficher les performances correspondantes
print("Résultats de la validation croisée :")
for mean, std, params in zip( #cv_results est un dictionnaire
        clf.cv_results_['mean_test_score'], # score moyen
        clf.cv_results_['std_test_score'],  # écart-type du score
        clf.cv_results_['params']           # valeur de l'hyperparamètre
    ):

    print("{} = {:.3f} error mean (+/-{:.03f})écart-type for {}".format(
        score,
        mean,
        std*2,
        params
    ) )


#regarder la performance sur le jeu de test. GridSearchCV a automatiquement ré-entraîné le meilleur modèle sur l’intégralité du jeu d’entraînement,
y_pred = clf.predict(X_test_std)#fonction prédict
#print("\nSur le jeu de test :  error mean = {:.3f}".format(metrics.accuracy_score(y_test, y_pred)))
print ("On va donc prendre le paramètre {}".format(clf.best_params_))

print("la perforamnce du modele avec cet hyperparamètre sur le jeu de test est de : {:.3f} . Soit {:.3f} des points sont biens classés  ".format(metrics.accuracy_score(y_test, y_pred),metrics.accuracy_score(y_test, y_pred)))


##Dessin de la courbe ROC
y_pred_proba = clf.predict_proba(X_test_std)[:, 1]
[fpr, tpr, thr] = metrics.roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='coral', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - specificite', fontsize=14)
plt.ylabel('Sensibilite', fontsize=14)
plt.show()
