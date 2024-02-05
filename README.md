# Multiple-ML-Model-Performance-Analysis.
About this project :
The objective of this project is to perform in depth Analysis of Diabetese Dataset, Evaluate the Performance of different ML Classification Model and to classify whether someone has diabetes or not.
Early Signs of Diabetes :
Hunger and fatigue. Your body converts the food you eat into glucose that your cells use for energy
Peeing more often and being thirstier
Dry mouth and itchy skin
Blurred vision

About the Dataset :
Pregnancies :- Number of times a woman has been pregnant
Glucose :- Plasma Glucose concentration of 2 hours in an oral glucose tolerance test
BloodPressure :- Diastollic Blood Pressure (mm hg)
SkinThickness :- Triceps skin fold thickness(mm)
Insulin :- 2 hour serum insulin(mu U/ml)
BMI :- Body Mass Index ((weight in kg/height in m)^2)
Age :- Age(years)
DiabetesPedigreeFunction :-scores likelihood of diabetes based on family history
Outcome :- 0 (doesn't have diabetes) or 1 (has diabetes)

Dataset Contains 768 data instances.
6 Independent Variables are of int64 datatype
2 Independent Variables are of float64 datatype
Dependent Variables is of int64 datatype
dependent variable has data imbalancy:
	- 65% data belongs to non-diabetic class
	- 35% data belongs to diabetic class
DataFrame is devoid of Missing values and Duplicated Rows.
DataFrame has few Outliers associated with some of the dependent variables
I will be checking Data Health (Missing Values, Duplicated Values, Statistical Summary, Outliers detection and Removal)
I will performing in depth EDA (Univariate, Bivariate and Multivariate Analysis)
I will perform the Hypothesis Testing to estabilish the relationship among Dependent and Independent Variables.

_______________________________________________________UNIVARIATE__________________
(1) Pregnancies : Number of times pregnant
	- Data is varing from 0 time of being pregnent  to 17 times being pregnent
	- Perform Feature Engineering , seperated the data on the bassis of number of pregnancies
	- Data is categorized in Low | High | Very-High band 
	- Low           45%
	- VeryHigh    29%
	- High           26%

(2) Glucose : Plasma glucose concentration a 2 hours in an oral glucose tolerance test
	-  Data has Skewness Coefficient of: 0.18 (It is not Skewed)
	- Skewness is because of Outliers(Data does not required any tranformation)
	- Most of the Population has glucose concentration between 90 - 120


(3) BloodPressure : Diastolic blood pressure (mm Hg)
		- BloodPressure has Skewness Coefficient of: -1.8 (Data seems Skewd to the left)	
		- Skewness is because of Outliers (Data does not require any tranformation)
		- Most of the Population has BloodPressure between 60 mm-hg - 80 mm-hg	


(4) SkinThickness : Triceps skin fold thickness (mm)
		SkinThickness has Skewness Coefficient of: 0.1 (Data is not shewd) 
		Skewness is because of Outliers (Data does not require any tranformation)
		Most of the Population has Skin fold thickness between 20 mm - 40 mm

(5) Insulin : 2-Hour serum insulin (mu U/ml)
		Insulin has Skewness Coefficient of: 2.2 (Data is heavily skewed toward right)
		Skewness is because of Outliers (Data does not require any tranformation)
		Most of the Population has Insulin level between 50(mu U/ml) - 125(mu U/ml)

(6) BMI : Body mass index (weight in kg/(height in m)^2)
	-BMI has Skewness Coefficient of: -0.4
	-Skewness is because of Outliers (Data does not require any tranformation)
	-Most of the Population has BMI ratio between 25  - 40 (weight in kg/(height in m)^2)


(7) DiabetesPedigreeFunction : Diabetes pedigree function
	-DiabetesPedigreeFunction has Skewness Coefficient of: 1.9 (Data is skewd toward right)
	-Skewness is because of Outliers (Data does not require any tranformation)
	-Most of the Population has Diabetes pedigree function between 20 - 40

(8) Age : Age (years)
	-Age has Skewness Coefficient of: 1.1(Data is skewed towards right)
	-Skewness is because of Outliers (Data does not require any tranformation)
	-Most of the Population has Age between 20  - 30 Years

(9) Outcome : Class variable (0 or 1)  1 (diabetic),  0 (non-diabetic)
	-Outcome is the Target Variable
	-Data has imbalancy
		-65% of Target data belongs to Class 0 (Non Diabetic)  
		-35% of Target data belongs to Class 1 (Diabetic) 
	-Data imbalacy would lead to biased learning (I am Using SMOTE technique to bring balancy in the data)   

 ____________________________________________________BIVARIATE________________________________________


(1) Outcome Vs  Pregnancies
Anova-Test : F_onewayResult(statistic=810.5150593469092, pvalue=1.7236475242343698e-143)
High F-static Score suggests strong relation, in this case, F-static score is low 
PointBiserial-Test : SignificanceResult(statistic=0.22189815303398686, pvalue=5.065127298053635e-10)
Score of F-static close to zero suggest no significant relations, close to 1 or -1 suggests relationship
Also from the plot, it is evident that there is no significant relations between Pregnancies and Outcome

(2) Outcome Vs  Glucose
Anova-Test : F_onewayResult(statistic=10914.672630134193, pvalue=0.0)
High F-static Score suggests strong relation, in this case, F-static score is High
PointBiserial-Test : SignificanceResult(statistic=0.466581398306874, pvalue=8.935431645289576e-43)
Score of F-static is not close to zero suggest some significant relations
Also from the plot, it is evident that there is significant difference among the mean of both classes, suggesting relations between Glucose and Outcome

(3) Outcome Vs  BloodPressure
Anova-Test : F_onewayResult(statistic=9685.068133631648, pvalue=0.0)
High F-static Score suggests strong relation, in this case, F-static score is considrably  High
PointBiserial-Test : SignificanceResult(statistic=0.06506835955033283, pvalue=0.07151390009776264)
Score of F-static is so close to zero suggest no any significant relations
Also from the plot, it is evident that there is not much  significant difference among the mean of both classes, suggesting no relation between BloodPressure and Outcome

(4) Outcome Vs  SkinThickness
Anova-Test : F_onewayResult(statistic=1228.8421887367274, pvalue=3.1079391612307788e-198)
High F-static Score suggests strong relation, in this case, F-static score is not  High
PointBiserial-Test : SignificanceResult(statistic=0.0747522319183194, pvalue=0.038347704820490915)
Score of F-static is so close to zero suggest no any significant relations
Also from the plot, it is evident that there is not much  significant difference among the mean of both classes, suggesting no relation between SkinThickness and Outcome

(5) Outcome Vs  Insulin 
Anova-Test : F_onewayResult(statistic=365.01491713877726, pvalue=3.6531836527047957e-73)
High F-static Score suggests strong relation, in this case, F-static score is not  High
PointBiserial-Test : SignificanceResult(statistic=0.13054795488404775, pvalue=0.0002861864603603164)
Score of F-static is so close to zero suggest no any significant relations
Also from the plot, it is evident that there is not much  significant difference among the mean of both classes, suggesting no relation between Insulin and Outcome

(6) Outcome Vs  BMI
Anova-Test : F_onewayResult(statistic=12326.397968979043, pvalue=0.0)
High F-static Score suggests strong relation, in this case, F-static score is   High
PointBiserial-Test : SignificanceResult(statistic=0.29269466264444544, pvalue=1.2298074873116917e-16)
Score of F-static is not so close to zero suggest some significant relations
Also from the plot, it is evident that there is not much but some  significant difference among the mean of both classes, suggesting some relation between BMI and Outcome

(7) Outcome Vs  DiabetesPedigreeFunction
Anova-Test : F_onewayResult(statistic=34.40531346539221, pvalue=5.471443802691407e-09)
High F-static Score suggests strong relation, in this case, F-static score is  Low
PointBiserial-Test : SignificanceResult(statistic=0.17384406565296007, pvalue=1.2546070101487771e-06)
Score of F-static is  close to zero suggest no significant relations
Also from the plot, it is evident that there is not much significant difference among the mean of both classes, suggesting no relation between  DiabetesPedigreeFunction and Outcome

(8) Outcome Vs Age
Anova-Test : F_onewayResult(statistic=5997.832961507344, pvalue=0.0)
High F-static Score suggests strong relation, in this case, F-static score is  considerably high
PointBiserial-Test : SignificanceResult(statistic=0.2383559830271977, pvalue=2.209975460665451e-11)
Score of F-static is not so  close to zero suggest some significant relations
Also from the plot, it is evident that there is  significant difference among the mean of both classes, suggesting some relation between  Age and Outcome

___CONCLUSION___
After an in-depth analysis of various machine learning models, including Logistic Regression, 
Support Vector Classifier (SVC), Decision Tree, Bagging, RandomForest, and Boosting, we have 
identified two standout performers for our task.

Decision Tree Model:
    Training Accuracy: 100%
    Validation Accuracy: 100%

Boosting ML Model:
    Training Accuracy: 98%
    Validation Accuracy: 100%

Observations:
The Decision Tree model demonstrates exceptional accuracy on both the training and validation datasets, 
achieving a perfect score of 100%.
The Boosting ML Model also exhibits outstanding performance, with a training accuracy of 98% and maintaining 
a flawless 100% validation accuracy.

Key Considerations:
The Decision Tree model excels in simplicity and interpretability, providing a robust solution with no signs of overfitting.
Boosting, with its ensemble approach, showcases impressive generalization capabilities, making it a reliable 
choice for accurate predictions.

Final Decision:
Based on the comprehensive evaluation of accuracy metrics and model behavior, 
both the Decision Tree and Boosting ML Models have proven to be highly effective. 
The choice between them may depend on specific project requirements, interpretability,
and the desired balance between simplicity and predictive power.



 

