/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.tensorflow;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.java.LayerTestBase;
import com.simiacryptus.mindseye.layers.java.MeanSqLossLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.util.TFConverter;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.tensorflow.GraphModel;
import com.simiacryptus.util.JsonUtil;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract class TFLayerTestBase extends LayerTestBase {

  private volatile @NotNull TFLayerBase tfLayer = null;

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    return getTfLayer();
  }

  public TFLayerBase getTfLayer() {
    if (null == tfLayer) {
      synchronized (this) {
        if (null == tfLayer) {
          TFLayerBase temp_01_0001 = createTFLayer();
          if (null != tfLayer)
            tfLayer.freeRef();
          tfLayer = temp_01_0001 == null ? null : temp_01_0001.addRef();
          if (null != temp_01_0001)
            temp_01_0001.freeRef();
        }
      }
    }
    return (TFLayerBase) tfLayer.copy();
  }

  public static @SuppressWarnings("unused") TFLayerTestBase[] addRefs(TFLayerTestBase[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TFLayerTestBase::addRef)
        .toArray((x) -> new TFLayerTestBase[x]);
  }

  public static @SuppressWarnings("unused") TFLayerTestBase[][] addRefs(TFLayerTestBase[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TFLayerTestBase::addRefs)
        .toArray((x) -> new TFLayerTestBase[x][]);
  }

  @Override
  public void run(@NotNull @Nonnull NotebookOutput log) {
    log.eval(() -> {
      TFLayerBase tfLayer = getTfLayer();
      GraphDef graphDef = tfLayer.constGraph();
      if (null != tfLayer)
        tfLayer.freeRef();
      GraphModel graphModel = new GraphModel(graphDef.toByteArray());
      return JsonUtil.toJson(graphModel);
    });
    super.run(log);
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    TFLayerBase tfLayer = getTfLayer();
    PipelineNetwork temp_01_0002 = new TFConverter().convert(tfLayer == null ? null : tfLayer.addRef());
    if (null != tfLayer)
      tfLayer.freeRef();
    return temp_01_0002;
  }

  public @SuppressWarnings("unused") void _free() {
    if (null != tfLayer)
      tfLayer.freeRef();
    tfLayer = null;
  }

  public @Override @SuppressWarnings("unused") TFLayerTestBase addRef() {
    return (TFLayerTestBase) super.addRef();
  }

  protected abstract @NotNull TFLayerBase createTFLayer();

  @Override
  protected Layer lossLayer() {
    return new MeanSqLossLayer();
  }
}
