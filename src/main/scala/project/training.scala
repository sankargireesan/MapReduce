package project


import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.mllib.classification.{LogisticRegressionModel,LogisticRegressionWithLBFGS}


import org.apache.spark.mllib.tree.model.{GradientBoostedTreesModel, RandomForestModel}
import org.apache.spark.mllib.tree.{GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, Strategy}

object training {
  
   def main(args: Array[String]) {
    
    val conf = new SparkConf()
               .setAppName("training")
//               .setMaster("local")

    val sc = new SparkContext(conf)
    
    println("start trainig set   :"+System.nanoTime())
    
    val trainingData =sc.textFile(args(0)+"/*.csv",sc.defaultParallelism)
            .map(record => record.split(","))
            .map(row =>     new LabeledPoint(
                                row.last.toDouble, 
                                Vectors.dense(row.take(row.length - 1).map(str => str.toDouble)))).cache()
//            .randomSplit(Array(0.34,0.33,0.33))
//              .randomSplit(Array(0.01,0.01,0.01,0.97))                 
            
//    val trainigDataM1 = trainingData(0).cache()
//    val trainigDataM3 = trainingData(1).cache()
//    val trainigDataM2 = trainingData(2).cache()
    
    val modelpath = args(1)
    
    println("end trainig set"+System.nanoTime())
    
    println("train models  :"+System.nanoTime())
    
    trainingGBT(sc,modelpath+"/GBTModel",trainingData)
    
//    sc.parallelize(
//        Array(trainingRF(sc,modelpath+"/RFModel",trainigDataM1),
//              trainingLR(sc,modelpath+"/LRModel",trainigDataM3),
//              trainingGBT(sc,modelpath+"/GBTModel",trainigDataM2)))

    println("end training models : "+System.nanoTime())
    
//    val rfModel = RandomForestModel.load(sc,modelpath+"/RFModel")    
//    
//    val lrModel =  LogisticRegressionModel.load(sc,modelpath+"/LRModel")
    
    val gbtModel =  GradientBoostedTreesModel.load(sc,modelpath+"/GBTModel")
    
//    val totCount = sc.accumulator(0,"count")
//    val rfCount = sc.accumulator(0,"rfcount")
//    val lrCount = sc.accumulator(0,"lrcount")
//    val gbtCount = sc.accumulator(0,"gbtcount")
    
    
    // validation
    println("validation begins : " +System.nanoTime())
    
    val validationData =sc.textFile(args(2)+"/L6_4_978344.csv")
            .map(record => record.split(","))
            .map(row =>     new LabeledPoint(
                                row.last.toDouble, 
                                Vectors.dense(row.take(row.length - 1).map(str => str.toDouble))))
                                    
    val testData1: RDD[LabeledPoint] = validationData.filter(r => r.label==1.0).cache()
    val testData0: RDD[LabeledPoint] = validationData.filter(r => r.label==0.0).cache()
    val test1 = testData1.count()
    val test0 = testData0.count()
     
                          
    var predictionAndLabels = testData1
            .filter(
              lp => predictGBT(lp, gbtModel) == lp.label)
             
    

    val predict = predictionAndLabels.count()
    println("predict 1 LR: "+predict)
    println("testData : "+test1)
    println("Accuracy LR= "+predict.toFloat/test1)    
    
    var predictionAndLabels0 = testData0
        .filter(
          lp => predictGBT(lp, gbtModel) == lp.label)
    

    val predict0 = predictionAndLabels0.count()
    println("predict 0 LR: "+predict0)
    println("testData : "+test0)
    println("Accuracy LR = "+predict0.toFloat/test0)    
    
    println("finish  :"+System.nanoTime())    
    
    println("total Accuracy :"+ (predict+predict0)/(test1+test0)) 
              
  }
  
   
 def calculatePrediction(
          lp: LabeledPoint,
          lrModel: LogisticRegressionModel,
          rfModel: RandomForestModel,
          gbtModel: GradientBoostedTreesModel): Int = {

      val predictions: Array[Int] =
          Array(
            predictLogisticRegression(lp, lrModel).toInt,
            predictGBT(lp, gbtModel).toInt,
            predictRandomForest(lp, rfModel).toInt)
     
      val sum = predictions.sum
      
      if(sum>=2)
       return 1
      else
        return 0
       
   
    }     
     
      def predictRandomForest(
          lp: LabeledPoint,
          rfModel: RandomForestModel) : Double = {
          return rfModel.predict(lp.features)
         
      }
      def predictLogisticRegression(
          lp: LabeledPoint,
          lrModel: LogisticRegressionModel) : Double = {
          return lrModel.predict(lp.features)
    }
      def predictGBT(
          lp: LabeledPoint,
          gbtModel: GradientBoostedTreesModel) : Double = {
          return gbtModel.predict(lp.features)
    }
      
   
   def trainingRF(sc: SparkContext,path: String, trainigDataM1: RDD[LabeledPoint])={
      println("training RF  :" +System.nanoTime())
      val treeStrategy = Strategy.defaultStrategy("Classification")
      val numTrees = 30
      val featureSubsetStrategy = "auto"
      val maxBins = 3000
      val numClasses = 2
      val categoricalFeaturesInfo = Map[Int, Int]()
      val maxDepth = 20
      val impurity = "gini"
      val rfModel= RandomForest
            .trainClassifier(trainigDataM1, 
                             numClasses, categoricalFeaturesInfo, numTrees, 
                             featureSubsetStrategy, impurity, maxDepth, maxBins)
      rfModel.save(sc,path)
      trainigDataM1.unpersist()
      println("training RF ends : "+System.nanoTime())
   }
     
   def trainingGBT(sc: SparkContext,path: String,trainigDataM2: RDD[LabeledPoint])={
      println("training GBT  :"+System.nanoTime())
      
      val numBoostingIterations = 3
      val maxBoostingDepth = 4
      val numClasses = 2
      
      var boostingStrategy = BoostingStrategy.defaultParams("Classification")
      boostingStrategy.setNumIterations(numBoostingIterations)
      boostingStrategy.treeStrategy.setNumClasses(numClasses)
      boostingStrategy.treeStrategy.setMaxDepth(maxBoostingDepth)
      
      GradientBoostedTrees.train(
              trainigDataM2,
              boostingStrategy).save(sc,path)
      trainigDataM2.unpersist()
      
      println("training GBT ends :"+System.nanoTime())
   }
   
   def trainingLR(sc: SparkContext,path: String,trainigDataM3: RDD[LabeledPoint])={
      println("training LR :"+System.nanoTime())
      val numClasses = 2
    
      val lrModel= new LogisticRegressionWithLBFGS()
                .setNumClasses(numClasses)
                .run(trainigDataM3)
      
      lrModel.save(sc, path)
      trainigDataM3.unpersist()
      println("training LR ends :"+System.nanoTime())
   }
 
}