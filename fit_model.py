from sklearn import linear_model
import numpy

#filename = "/home/ekhnaton/Documents/Inka/housing.csv"
#filename = "/home/muti/Downloads/housing.csv"
filename = '/static/housing.csv'
data=numpy.genfromtxt(filename,delimiter=',',dtype=float,skip_header=1)
features = data[:,0:5]
targets = data[:,5]

# extract variables to correct format

model = linear_model.LinearRegression()
model.fit(features, targets) # array of input variables, array of output variables
model.coef_ # contains the coefficients
model.intercept_
model_coef_with_intercept = numpy.append(model.coef_,model.intercept_)

# save model
#output_filename = "/home/ekhnaton/Documents/Inka/housing_model.txt"
#output_filename = "/home/muti/Documents/housing_model.txt"
output_filename = '/static/housing_model.txt'
with open(output_filename, 'ab') as outfile:
    numpy.savetxt(outfile, model_coef_with_intercept)

# check results / do cross-validation tjsp.

