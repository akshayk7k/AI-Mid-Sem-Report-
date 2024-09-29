install.packages("bnlearn")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("e1071")




course.grades<-read.table("2020_bn_nb_data.txt",head=TRUE)
head(course.grades)


#printing baysian network


library(bnlearn)
course.grades<- lapply(course.grades,as.factor)
course.grades<- data.frame(course.grades)
course.grades.net<- hc(course.grades[,-9],score = 'k2')
plot(course.grades.net)

course.grades.net

course.grades.fit <- bn.fit(course.grades.net,course.grades[,-9])
course.grades.fit

for (i in seq_along(course.grades.fit)) {
    bn.fit.barchart(course.grades.fit[[i]])
}


course.grades.PH100 <- data.frame( cpdist(course.grades.fit, nodes = c("PH100"),evidence = (EC100 == "DD") & (IT101 == "CC") & (MA101 == "CD")))
# course.grades.PH100

library(dplyr)
# Predicting the grade in PH100 based on evidence from EC100, IT101, and MA101
df <- course.grades.PH100 %>%
  group_by(PH100) %>%
  summarise(counts = n())

library(ggplot2)
ggplot(df, aes(x = PH100, y = counts)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = counts), vjust = -0.3)


  
library(e1071)

# Read the dataset
data <- read.table("2020_bn_nb_data.txt", header=TRUE)

# Set the seed for reproducibility
set.seed(123)

# Number of iterations for cross-validation
num_iterations <- 20

# Initialize vector to store accuracy results
accuracy_results <- numeric(num_iterations)

# Perform cross-validation
for (i in 1:num_iterations) {
  # Shuffle the data
  shuffled_data <- data[sample(nrow(data)), ]

  # Calculate the index to split data into training and testing sets (70% training, 30% testing)
  split_index <- as.integer(0.7 * nrow(shuffled_data))

  # Split the data into training and testing sets
  training_data <- shuffled_data[1:split_index, ]
  testing_data <- shuffled_data[(split_index + 1):nrow(shuffled_data), ]


  nb_classifier <- naiveBayes(QP ~ ., data = training_data, laplace = 1)

  predictions <- predict(nb_classifier, testing_data[, -ncol(testing_data)], type = "class")


  accuracy <- mean(predictions == testing_data$QP)


  accuracy_results[i] <- accuracy
}

average_accuracy <- mean(accuracy_results)
cat("Average Accuracy ", average_accuracy, "\n")




library(e1071)

# Read the data from the provided file
course.grades <- read.table("2020_bn_nb_data.txt", head = TRUE)

# Initialize vector to store accuracies
accuracies <- numeric(20)

# Repeat 20 random selections of training and testing data
for (i in 1:20) {
  set.seed(i)  # Set seed for reproducibility
  train_indices <- sample(1:nrow(course.grades), 0.7 * nrow(course.grades))
  train_data <- course.grades[train_indices, ]
  test_data <- course.grades[-train_indices, ]

  # Build naive Bayes classifier using training data
  model <- naiveBayes(QP ~ ., data = train_data)
  predictions <- predict(model, test_data)
  accuracies[i] <- mean(predictions == test_data$QP)
  cat("Accuracy for iteration", i, ":", round(accuracies[i] * 100, 2), "%\n")}

average_accuracy <- mean(accuracies)  # Calculate average accuracy
cat("Average Accuracy:", round(average_accuracy * 100, 2), "%\n") #ave accuracy