package com.tencent.angel.graph.embedding.deepwalk

import com.tencent.angel.graph.common.param.ModelContext
import com.tencent.angel.graph.common.psf.param.LongKeysUpdateParam
import com.tencent.angel.graph.common.psf.result.GetLongsResult
import com.tencent.angel.graph.model.general.init.GeneralInit
import com.tencent.angel.graph.utils.ModelContextUtils
import com.tencent.angel.ml.matrix.RowType
import com.tencent.angel.ps.storage.vector.element.IElement
import com.tencent.angel.spark.ml.util.LoadBalancePartitioner
import com.tencent.angel.spark.models.PSMatrix
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap
import org.apache.spark.rdd.RDD
import com.tencent.angel.graph.psf.neighbors.SampleNeighborsWithCount.{GetNeighborWithCountParam, GetNeighborsWithCount, NeighborsAliasTableElement}

class DeepWalkPSModel(val edgesPsMatrix: PSMatrix) extends Serializable {
  //push node adjacency list

  def initNodeNei(msgs: Long2ObjectOpenHashMap[NeighborsAliasTableElement]): Unit = {
    val nodeIds = new Array[Long](msgs.size())
    val neighborElems = new Array[IElement](msgs.size())
    val iter = msgs.long2ObjectEntrySet().fastIterator()
    var index = 0
    while (iter.hasNext) {
      val i = iter.next()
      nodeIds(index) = i.getLongKey
      neighborElems(index) = i.getValue
      index += 1
    }

    edgesPsMatrix.psfUpdate(new GeneralInit(new LongKeysUpdateParam(edgesPsMatrix.id, nodeIds, neighborElems))).get()
  }


  //pull node adjacency list
  def getSampledNeighbors(psMatrix: PSMatrix, nodeIds: Array[Long], count: Array[Int]): Long2ObjectOpenHashMap[Array[Long]] = {
    psMatrix.psfGet(
      new GetNeighborsWithCount(
        new GetNeighborWithCountParam(psMatrix.id, nodeIds, count))).asInstanceOf[GetLongsResult].getData
  }

  def checkpoint(): Unit = {
    edgesPsMatrix.checkpoint()
  }
}

object DeepWalkPSModel {
  def apply(modelContext: ModelContext, data: RDD[Long],
            useBalancePartition: Boolean, balancePartitionPercent: Float): DeepWalkPSModel = {
    val matrix = ModelContextUtils.createMatrixContext(modelContext, RowType.T_ANY_LONGKEY_SPARSE, classOf[NeighborsAliasTableElement])

    // TODO: remove later
    if (!modelContext.isUseHashPartition && useBalancePartition)
      LoadBalancePartitioner.partition(
        data, modelContext.getMaxNodeId, modelContext.getPartitionNum, matrix, balancePartitionPercent)

    val psMatrix = PSMatrix.matrix(matrix)
    new DeepWalkPSModel(psMatrix)

  }
}