# atos
importing mixed data types and using pandas, svm and decision tree for decision.
Build a solution in the programming language of (Python ) that predicts if the patients will suffer from stroke or not (use 70:30 split between your train and test datasets). Thanks to that the hospital can identify patients with high risk of stroke and sent them for proper treatment
We dont have reliable, long data which can be used for prediction of stroke. Data is unbalanced. No information of past history. I used a simple assumptions that first variables contain information which can be used to predict last variable. This is very simplistic- we should have time series data ,monitoring state of patient, knowledge of the predicive variables of state of patient before stroke and more relavent variables. We can see confusion matrix to verify the results. I used 2 methods, svm and decision tree.

Atos – Stroke prediction
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
I  had access to only very limited data sets.  Question arises if data science is the right approach to solve that problem.
As  in any task we have 3 main components:
1, Assumptions: These are taken from our experience and intuition to be the basis of our thinking about a problem.
2, Model: This is the representation of our assumptions in a way that we can reason (i.e. as an equation or a simulation). I utilized very simplistic model , fit for the data.
3, Data: This is what we measure and understand about the real world. Questions- How good are data?  When they were measured? etc

Tools used- importing mixed data types and using pandas, svm and decision tree for decision about stroke. 
Build a solution in the programming language of (Python ) that predicts if the patients will suffer from stroke or not (use 70:30 split between your train and test datasets). Thanks to that the hospital can identify patients with high risk of stroke and sent them for proper treatment We dont have reliable, long data which can be used for prediction of stroke. Data is unbalanced. No information of past history. I used a simple assumptions that first variables contain information which can be used to predict last variable. This is very simplistic- we should have time series data , monitoring state of patient, knowledge of the predictive variables of state of patient before stroke and more relevant variables. We can see confusion matrix to verify the results. I used 2 methods, svm and decision tree.
It raised serious issues around machine learning approaches to a wide-range of problems. It is an example of modeling to illustrate my concerns. Both models ( decision tree and svm) are equally good, according to the algorithm, but they make similar predictions about the eventual strike.
This is an example of what is known as an underspecification problem: many models explain the same data. Uderspecification presents significant challenges for the credibility of modern machine learning. It affects everything from stroke detection to any other complex models.
Often there are a whole range of models that might explain a particular data set: how can we identify which one is best? The answer is that we can’t. Or at least we can’t do so without making and documenting further assumptions. If we want to model stroke, we make assumptions about peoples behaviour — how they will respond to many other factors, not measured here, etc. We know that not everything we assume can be measured. Assumptions thus document what we don’t know, but think might be the case.
This is just one model of many, possibly infinite many, alternatives. It is one way of looking at the world. In emphasizing the model researchers persuing a pure machine learning approach make a very strong implicit assumption: that their model doesn’t need assumptions.  All models need assumptions.

