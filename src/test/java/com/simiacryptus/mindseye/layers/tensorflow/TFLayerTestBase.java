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
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.util.TFConverter;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.tensorflow.GraphModel;
import com.simiacryptus.util.JsonUtil;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public abstract class TFLayerTestBase extends LayerTestBase {

  @Override
  public void run(@Nonnull NotebookOutput log) {
    log.eval(() -> {
      TFLayerBase tfLayer = getTfLayer();
      GraphDef graphDef = tfLayer.constGraph();
      GraphModel graphModel = new GraphModel(graphDef.toByteArray());
      try {
        return JsonUtil.toJson(graphModel);
      } finally {
        tfLayer.freeRef();
      }
    });
    super.run(log);
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    TFLayerBase tfLayer = getTfLayer();
    DAGNetwork convert = new TFConverter().convert(tfLayer);
    tfLayer.freeRef();
    return convert;
  }

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    return getTfLayer();
  }

  protected abstract @NotNull TFLayerBase createTFLayer();

  @Override
  protected Layer lossLayer() {
    return new MeanSqLossLayer();
  }

  private volatile @NotNull TFLayerBase tfLayer = null;
  public TFLayerBase getTfLayer() {
    if(null == tfLayer) {
      synchronized (this) {
        if(null == tfLayer) {
          tfLayer = createTFLayer();
        }
      }
    }
    return (TFLayerBase) tfLayer.copy();
  }
}
