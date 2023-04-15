import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StructField, StructType,IntegerType}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.MulticlassMetrics


val spark = SparkSession.builder().getOrCreate()


var data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("framingham_heart_disease.csv")
println(s"Number of Rows in Dataset : ${data.count()}")
println(s"Number of Columns in Dataset : ${data.columns.length()}")
// Droped Rows with NA Values
data = data.filter(not($"glucose".equalTo("NA")) && not($"cigsPerDay".equalTo("NA")) && not($"BPMeds".equalTo("NA")) && not($"totChol".equalTo("NA")) && not($"BMI".equalTo("NA")) &&  not($"heartRate".equalTo("NA")))
data.show()
println("Removed rows which have empty values in their columns")
println("After Removing them : ")
println(s"Number of Rows in Dataset : ${data.count()}")
println(s"Number of Columns in Dataset : ${data.columns.length()}")

println("Dataset Schema : ")
data.printSchema()
// Need to typecast columns having string as data type as they 
data = data.withColumn("cigsPerDay", col("cigsPerDay").cast(DoubleType))
data = data.withColumn("BPMeds", col("BPMeds").cast(DoubleType))
data = data.withColumn("totChol", col("totChol").cast(DoubleType))
data = data.withColumn("BMI", col("BMI").cast(DoubleType))
data = data.withColumn("heartRate", col("heartRate").cast(DoubleType))
data = data.withColumn("glucose", col("glucose").cast(DoubleType))
data.printSchema()
val logregdataall = data.select(data("TenYearCHD").as("label") ,$"male", $"age", $"currentSmoker",$"cigsPerDay", $"BPMeds", $"prevalentStroke", $"prevalentHyp", $"diabetes", $"totChol", $"sysBP", $"diaBP", $"BMI", $"heartRate", $"glucose")
val logregdata = logregdataall.na.drop()

/*
    Random Splitting of data
        70% of data is being used for training
        30% of data is being used for testing the model
*/
val Array(training,test) = logregdata.randomSplit(Array(0.7,0.3),seed=123)

// Decalaring the assempler for the model
val assembler = new VectorAssembler().setInputCols(Array("male", "age", "currentSmoker","cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose")).setOutputCol("features")


// Declared Logistic Regression Model 
val lr = new LogisticRegression() 

val pipeline = new Pipeline().setStages(Array(assembler,lr))

val model = pipeline.fit(training)
/* For saving the model we can remove comment from below 2 lines then it would store model in model directory 
(Note : We have to first make sure to not have any ./model directory before running this file otherwise this will give an error)
*/


// println("Saving the model : ")
// model.save("./model")

val results = model.transform(test)

val predictionAndLabels = results.select($"prediction",$"label").as[(Double,Double)].rdd

val metrics = new MulticlassMetrics(predictionAndLabels)


println("Model Evaluation : ")
println("Cofusion Matix : ")

// Calculate metrics values individually
// Assigning Variables for convinience

val confusionMatrix = metrics.confusionMatrix

val TN =  confusionMatrix.apply(0,0)
val FP =  confusionMatrix.apply(0,1)
val FN =  confusionMatrix.apply(1,0)
val TP =  confusionMatrix.apply(1,1)
println("Confusion Matrix Values : ")
println(s"TN : ${TN}\nFP : ${FP}\nFN : ${FN}\nTP : ${TP}")

//  Calculating precision, recall, specificity, and accuracy mathematically by using their equations 

val recall = TP / (TP + FN )
println(s"Recall Value : ${recall}")

val precision = TP / (TP + FP)
println(s"Precision Value : ${precision}")

val specificity = TN /  (TN + FP)
println(s"Specificity Value : ${specificity}")

val accuracy = ( TP + TN ) / ( TP + TN + FP + FN)
print(s"Model Accuracy : ${accuracy} ( ${accuracy*100} %)")




