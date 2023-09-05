/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */

package com.tencent.angel.graph.community.lpa

import com.tencent.angel.graph.common.param.ModelContext
import com.tencent.angel.graph.utils.io.Log
import com.tencent.angel.graph.utils.{GraphIO, Stats}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.graph.utils.params._
import org.apache.spark.SparkContext
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

class LPA(override val uid: String) extends Transformer
  with HasSrcNodeIdCol with HasDstNodeIdCol with HasOutputNodeIdCol with HasOutputCoreIdCol
  with HasStorageLevel with HasPartitionNum with HasPSPartitionNum with HasUseBalancePartition
  with HasBalancePartitionPercent with HasNeedReplicaEdge with HasMaxIteration with HasBatchSize {
  
  def this() = this(Identifiable.randomUID("LPA"))
  
  override def transform(dataset: Dataset[_]): DataFrame = {
    val edges = GraphIO.loadEdgesFromDF(dataset, $(srcNodeIdCol), $(dstNodeIdCol))  // 读取边数据
    edges.persist($(storageLevel))
    
    val (minId, maxId, numEdges) = Stats.summarize(edges)  // 统计数据 最小id， 最大id ， 边个数
    Log.withTimePrintln(s"minId=$minId maxId=$maxId numEdges=$numEdges level=${$(storageLevel)}")
    
    // Start PS and init the model
    Log.withTimePrintln("start to run ps")
    PSContext.getOrCreate(SparkContext.getOrCreate())

    // 构造lpa的模型
    val modelContext = new ModelContext($(psPartitionNum), minId, maxId, -1,
      "lpa", SparkContext.getOrCreate().hadoopConfiguration)
    val model = LPAPSModel(modelContext, edges, $(useBalancePartition), $(balancePartitionPercent))
  
    val loadGraphTime = System.currentTimeMillis()
    // 这里看不懂  看逻辑是反向一下， 使图变为无向图  但是命名却是备份
    val newEdges = if ($(needReplicaEdge)) edges.flatMap(f => Iterator((f._1, f._2), (f._2, f._1))) else edges

    val graph = newEdges
      .groupByKey($(partitionNum))
      .mapPartitionsWithIndex((index, it) =>
        Iterator(LPAGraphPartition.apply(index, it)))
    
    graph.persist($(storageLevel))
    graph.foreachPartition(_ => Unit) // action 开始执行，准备压缩邻接表数据
    println(s"make graph partitions cost: ${(System.currentTimeMillis() - loadGraphTime) / 1000.0} s")
  
    graph.foreach(_.initMsgs(model, $(batchSize)))
    edges.unpersist(blocking = false)
  
    var curIteration = 0
    val maxIterNum = $(maxIteration)
    var changedNum = 0L

    do {
      val iterationTime = System.currentTimeMillis()
      curIteration += 1
      changedNum = graph.map(_.process(model, $(batchSize))).reduce(_ + _)
      model.resetMsgs()
      Log.withTimePrintln(s"LPA finished iteration $curIteration; $changedNum  nodes changed lpa label, " +
        s"cost: ${(System.currentTimeMillis() - iterationTime) / 1000.0} s")
    } while (curIteration < maxIterNum && changedNum != 0)

    val retRDD = graph.flatMap(_.save(model, $(batchSize))) // 需要从angel中取出来结果
      .sortBy(_._2)
      .map(f => Row.fromSeq(Seq[Any](f._1, f._2)))

    
    dataset.sparkSession.createDataFrame(retRDD, transformSchema(dataset.schema))
  }
  
  override def transformSchema(schema: StructType): StructType = {
    StructType(Seq(
      StructField(s"${$(outputNodeIdCol)}", LongType, nullable = false),
      StructField(s"${$(outputCoreIdCol)}", LongType, nullable = false)
    ))
  }
  
  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
  
}