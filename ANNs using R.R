library(data.table)
library(caret)
library(caTools)


#===============================================================#
#       DATA PRE-PROCESSING
#===============================================================#

# Read in the data
#-----------------
dataset = fread('C:/Users/Amit/Desktop/Python Programming/Data/Artificial_Neural_Networks/Churn_Modelling.csv')
head(dataset)

# Remove redundant variables
#---------------------------
dataset = dataset[,-1:-3]

# Do one-hot encoding (this converts to a matrix, hence reconverting back to a data.table)
#-----------------------------------------------------------------------------------------
dummy = dummyVars(data=dataset, ~., fullRank=TRUE)
dataset = as.data.frame(predict(dummy, dataset))

# Scaling the data
#-----------------
dataset[,-12] = scale(dataset[,-12])
head(dataset)

# Split into training and test sets
#----------------------------------
set.seed(100)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
train = dataset[split==TRUE,]
test = dataset[split==FALSE,]

dim(train)
dim(test)

#===============================================================#
#       BUILDING THE NEURAL NETWORK
#===============================================================#

# We will use the "H2O" package. It's oneof the best available, and offers lots of options
# for cross valiation and fine tuning the model

library(h2o)

# Let's initialize the H2O environment. It creates a virtual environment and then runs the 
# neural network on that virtual server

h2o.init(nthreads = -1)   # -1 uses all the cores available in the system

# Now we'll build the actual model
classifier = h2o.deeplearning(y = 'Exited',
                              training_frame = as.h2o(train),   # we need to convert the training set to a H2O object
                              activation='Rectifier',  # using a Rectified Linear Unit for activation
                              hidden=c(6,6),   # 2 hidden layers with 6 neurons each
                              epochs = 100,    # number of iterations of the full dataset
                              train_samples_per_iteration = -2) #-2 will auto-tune


#===============================================================#
#       EVALUATING THE MODEL
#===============================================================#

# Predict the probabilities on the testing set
pred_prob = predict(classifier, as.h2o(test))

# Convert the probabilities to a binary output 1/0
y_pred = as.vector(pred_prob > 0.5)

# Create the confusion matrix to evaluate the model
confusionMatrix(test$Exited, y_pred)

# We are getting an accuracy of over 85% which is very good!!!


# Finally we shut down the H2O environment
h2o.shutdown()

