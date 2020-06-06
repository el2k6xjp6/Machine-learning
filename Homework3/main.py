import SequentialEstimator
import BayesianLinearRegression

estimator=SequentialEstimator.Estimator(3,5)
estimator.Estimate()
input("")
regression=BayesianLinearRegression.Regression(1,4,1,[1,2,3,4])
regression.Run()
regression=BayesianLinearRegression.Regression(100,4,1,[1,2,3,4])
regression.Run()
regression=BayesianLinearRegression.Regression(1,3,1/3,[1,2,3])
regression.Run()