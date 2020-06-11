"""
DOCSTRING
"""
import matplotlib.pyplot as pyplot
import numpy
import pandas
import sklearn.manifold as manifold
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing

dataframe_all = pandas.read_csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv')

num_rows = dataframe_all.shape[0]

counter_nan = dataframe_all.isnull().sum()

counter_without_nan = counter_nan[counter_nan==0]

dataframe_all = dataframe_all[counter_without_nan.keys()]

dataframe_all = dataframe_all.iloc[:,7:]

columns = dataframe_all.columns

print(columns)

x = dataframe_all.iloc[:,:-1].values

standard_scaler = preprocessing.StandardScaler()

x_std = standard_scaler.fit_transform(x)

y = dataframe_all.iloc[:,-1].values

class_labels = numpy.unique(y)

label_encoder = preprocessing.LabelEncoder()

y = label_encoder.fit_transform(y)

test_percentage = 0.1

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x_std, y, test_size = test_percentage, random_state = 0)

tsne = manifold.TSNE(n_components=2, random_state=0)

x_test_2d = tsne.fit_transform(x_test)

markers=('s', 'd', 'o', '^', 'v')

color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan'}

pyplot.figure()

for idx, cl in enumerate(numpy.unique(y_test)):
    pyplot.scatter(
        x=x_test_2d[y_test==cl,0],
        y=x_test_2d[y_test==cl,1],
        c=color_map[idx],
        marker=markers[idx],
        label=cl)

pyplot.xlabel('X in t-SNE')
pyplot.ylabel('Y in t-SNE')
pyplot.legend(loc='upper left')
pyplot.title('t-SNE visualization of test data')
pyplot.show()
