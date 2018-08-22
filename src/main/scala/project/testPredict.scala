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

object testPredict {
  def main(args: Array[String]) {
      val conf = new SparkConf()
               .setAppName("training")
               .setMaster("local")

    val sc = new SparkContext(conf)
      val modelpath = args(0)
    val rfModel = LogisticRegressionModel.load(sc,modelpath+"/LRModel")    
    
    // validation
    println("validation begins : " +System.nanoTime())
    
    val validationData =sc.textFile(args(1)+"/L6_4_978344.csv")
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
              lp => rfModel.predict(lp.features).toDouble == lp.label)
             
    

    val predict = predictionAndLabels.count()
    println("predict 1 LR: "+predict)
    println("testData : "+test1)
    println("Accuracy LR= "+predict.toFloat/test1)    
    
    var predictionAndLabels0 = testData0
        .filter(
              lp => rfModel.predict(lp.features).toDouble == lp.label)
    

    val predict0 = predictionAndLabels0.count()
    println("predict 0 RF: "+predict0)
    println("testData : "+test0)
    println("Accuracy RF = "+predict0.toFloat/test0)    
    
    println("finish  :"+System.nanoTime())    
    
    println("total Accuracy :"+ (predict+predict0).toFloat/(test1+test0)) 
  }
}