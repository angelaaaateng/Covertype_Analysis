# Covertype_Analysis
Repo that explores various forest covertypes and houses a flask multi-class classifier

This project shows the process of analyzing forest cover types from a dataset provided by the US Forest Service, and housed on the UCI Machine Learning Repository.

This dataset is specifically interesting because it consists of a mix of both categorical and continuous variables, which has historically required different techniques of analysis. These variables describe the geology of each sample forest region, and a multiclass label (one of seven possible tree cover types) serves as our target variable.

These seven possible cover types are as follows:

1. Spruce/Fir
2. Lodgepole Pine
3. Ponderosa Pine
4. Cottonwood/Willow
5. Aspen
6. Douglas-fir
7. Krummholz

Successful forest cover type classification has so much potential for positive change, particularly in areas like environmental conservation, flora and fauna research, and geological studies.

The dataset was retrieved from this source: https://archive.ics.uci.edu/ml/datasets/covertype

The flask app can be found here: https://covertype.herokuapp.com/ When testing out a dataset, ensure that it has at least one instance of each class.

The blog post that describes this can be found here:

Important Heroku commands for commiting + running on local: 
git remote -v 
git add . 
git commit -am " " 
git push heroku master
heroku ps:scale web=1
heroku logs --tail
heroku open 
