from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import numpy

#filename = "/home/ekhnaton/Documents/Inka/housing.csv"
#filename = "/home/muti/Downloads/housing.csv"
filename = '/app/housing.csv'
data=numpy.genfromtxt(filename,delimiter=',',dtype=float,skip_header=1)
features = data[:,0:5]
targets = data[:,5]

model = linear_model.LinearRegression(normalize=True)

# check that the model looks ok
mean_error = [abs(numpy.mean(cross_val_score(model,features,targets,cv=10,scoring='neg_mean_absolute_error')))]
# mean error is 4.29, target range 5-50 => ok


model.fit(features, targets) # array of input variables, array of output variables
model_coef_with_intercept = numpy.append(model.coef_,model.intercept_) # include intercept


# save model
#output_filename = "/home/ekhnaton/Documents/Inka/housing_model.txt"
#output_filename = "/home/muti/Documents/housing_model.txt"
output_filename = '/app/housing_model.txt'
with open(output_filename, 'ab') as outfile:
    numpy.savetxt(outfile, model_coef_with_intercept)

# save mean error
#output_filename2 = "/home/muti/Documents/model_error.txt"
output_filename2 = '/app/model_error.txt'
with open(output_filename2, 'ab') as outfile:
    numpy.savetxt(outfile, mean_error)


