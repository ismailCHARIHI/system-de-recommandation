##########################################################
# Créer un ensemble edx, un ensemble de validation (ensemble de test final)
##########################################################

# Note : ce processus pourrait prendre quelques minutes

# Vérifier et installer les packages nécessaires
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# Charger les bibliothèques nécessaires
library(tidyverse)
library(caret)
library(data.table)

# Jeu de données MovieLens 10M :
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Télécharger le fichier zip
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Lire les données de notation (ratings) depuis le fichier et les stocker dans un objet data.table
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# Lire les données des films (movies) depuis le fichier et les stocker dans un objet data.frame
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# Si vous utilisez R 4.0 ou une version ultérieure, convertir movies en data.frame
movies <- as.data.frame(movies) %>% 
  mutate(movieId = as.numeric(movieId),
         title = as.character(title),
         genres = as.character(genres))

# Fusionner les données de notation (ratings) et les données des films (movies) en utilisant movieId comme clé
movielens <- left_join(ratings, movies, by = "movieId")

# L'ensemble de validation sera de 10% des données MovieLens
set.seed(1, sample.kind="Rounding")  # Si vous utilisez R 3.5 ou une version antérieure, utilisez `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# S'assurer que userId et movieId dans l'ensemble de validation sont également présents dans l'ensemble edx
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Ajouter les lignes supprimées de l'ensemble de validation à nouveau dans l'ensemble edx
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Nettoyer les objets inutiles
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Charger les bibliothèques nécessaires pour la suite de l'analyse
library(ggplot2)
library(lubridate)

# Prétraitement des données
# Modifier l'année comme une colonne dans les deux ensembles de données
edx <- edx %>% mutate(year = as.numeric(str_sub(title, -5, -2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title, -5, -2)))

# Fonction de perte d'erreur quadratique moyenne (RMSE)
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = TRUE))
}

# Analyse exploratoire des données
head(edx) 

# Statistiques sommaires de l'ensemble de données edx
summary(edx)

# Nombre de films et d'utilisateurs uniques dans l'ensemble de données edx 
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

# Les évaluations des films sont dans chacun des genres suivants dans l'ensemble de données edx
genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

# Statistiques sommaires des évaluations dans edx
summary(edx$rating)

# Film ayant le plus grand nombre d'évaluations
edx %>% group_by(title) %>% summarize(number = n()) %>% arrange(desc(number))

# Les cinq évaluations les plus données, de la plus élevée à la moins élevée
head(sort(-table(edx$rating)), 5)

# Tracer des évaluations 
table(edx$rating)
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()

# À partir du tracé ci-dessus, on peut observer que les évaluations à demi-étoile sont moins courantes que les évaluations entières

# Évaluations moyennes de l'ensemble de données edx
avg_ratings <- edx %>% group_by(year) %>% summarize(avg_rating = mean(rating)) 

# Stratégies d'analyse des données
## À partir de l'analyse exploratoire des données ci-dessus, nous pouvons modéliser les effets Utilisateur, Film, Année

# Histogramme des évaluations
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "black") +
  xlab("Évaluation") +
  ylab("Nombre") +
  ggtitle("Histogramme des évaluations") +
  theme(plot.title = element_text(hjust = 0.5))     

## Table des 20 films évalués une seule fois
# Ces estimations bruyantes peuvent augmenter notre RMSE
edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  filter(count == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = rating, n_rating = count) %>%
  slice(1:20) %>%
  knitr::kable()

# Distribution des évaluations de chaque utilisateur pour les films - Effet de l'utilisateur
edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Biais des utilisateurs")

# Distribution des biais de film car la plupart des films à succès sont fortement évalués - Effet du film
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Biais des films")

# Estimation de la tendance de l'évaluation par rapport à l'année de sortie - Effet de l'année
edx %>% group_by(year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  geom_smooth()

# Création du modèle
# Modèle final
# Régression régularisée en utilisant les films, les utilisateurs, les années 
## Approche des moindres carrés pénalisés [minimiser une équation qui ajoute une pénalité]
# lambda est un paramètre de réglage. Utilisation de la validation croisée pour trouver la valeur optimale de lambda
## Veuillez noter que le code ci-dessous pourrait prendre un certain temps
# b_i, b_u, b_y représentent respectivement les effets du film, de l'utilisateur, de l'année
lambdas <- seq(0, 5, 0.25)

rmses <- sapply(lambdas, function(l) {
  
  # Calculer la moyenne des évaluations à partir de l'ensemble d'entraînement edx
  mu <- mean(edx$rating)
  
  # Ajuster la moyenne par l'effet du film et pénaliser les faibles nombres d'évaluations
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + l))
  
  # Ajuster la moyenne par l'effet de l'utilisateur et du film et pénaliser les faibles nombres d'évaluations
  b_u <- edx %>% 
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n() + l))
  
  # Ajuster la moyenne par l'effet de l'utilisateur, du film et de l'année et pénaliser les faibles nombres d'évaluations
  b_y <- edx %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u)/(n() + l), n_y = n())
  
  # Prédire les évaluations dans l'ensemble d'entraînement pour trouver la valeur optimale de la pénalité 'lambda'
  predicted_ratings <- edx %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  # Retourner l'erreur quadratique moyenne entre les évaluations réelles et prédites
  return(RMSE(edx$rating, predicted_ratings))
})

# Tracer la relation entre lambdas et rmses
plot(lambdas, rmses)

# Sélectionner la valeur optimale de lambda qui minimise l'erreur
lambda <- lambdas[which.min(rmses)]

# Appliquer lambda sur l'ensemble de validation
mu <- mean(edx$rating)
movie_effect_reg <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n() + lambda), n_i = n())
user_effect_reg <- edx %>% 
  left_join(movie_effect_reg, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n() + lambda), n_u = n())
year_reg_avgs <- edx %>%
  left_join(movie_effect_reg, by = "movieId") %>%
  left_join(user_effect_reg, by = "userId") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u)/(n() + lambda), n_y = n())
# Prédire les évaluations sur l'ensemble de validation
predicted_ratings <- validation %>% 
  left_join(movie_effect_reg, by = "movieId") %>%
  left_join(user_effect_reg, by = "userId") %>%
  left_join(year_reg_avgs, by = 'year') %>%
  mutate(pred = mu + b_i + b_u + b_y) %>% 
  .$pred
model_rmse <- RMSE(validation$rating, predicted_ratings)

# Résultat
rmse_results <- data_frame(method = "Modèle Effet Film, Utilisateur, Année Régularisé")
                           
                           