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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.tensorflow.GraphModel;
import com.simiacryptus.util.JsonUtil;
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

public abstract class TFLayerTestBase extends LayerTestBase {

  private volatile @Nonnull
  TFLayerBase tfLayer = null;

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    return getTfLayer();
  }

  @Nonnull
  public TFLayerBase getTfLayer() {
    return (TFLayerBase) tfLayer.copy();
  }

  @Override
  public void run(@Nonnull NotebookOutput log) {
    log.eval(() -> {
      TFLayerBase tfLayer = getTfLayer();
      GraphDef graphDef = tfLayer.constGraph();
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
    PipelineNetwork temp_01_0002 = new TFConverter().convert(tfLayer.addRef());
    tfLayer.freeRef();
    return temp_01_0002;
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
    tfLayer.freeRef();
    tfLayer = null;
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  TFLayerTestBase addRef() {
    return (TFLayerTestBase) super.addRef();
  }

  protected abstract @Nonnull
  TFLayerBase createTFLayer();

  @Nonnull
  @Override
  protected Layer lossLayer() {
    return new MeanSqLossLayer();
  }
}
