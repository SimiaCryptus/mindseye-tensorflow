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
import com.simiacryptus.mindseye.layers.java.MeanSqLossLayer;
import com.simiacryptus.mindseye.test.LayerTestBase;
import com.simiacryptus.mindseye.util.TFConverter;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.tensorflow.GraphModel;
import com.simiacryptus.util.JsonUtil;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.concurrent.TimeUnit;

public abstract class TFLayerTestBase extends LayerTestBase {

  private final @Nonnull
  @RefIgnore
  TFLayerBase tfLayer = createTFLayer();

  @Nonnull
  @Override
  public Layer getLayer() {
    return new TFConverter().convert(getTfLayer());
  }

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    return getTfLayer();
  }

  @Nonnull
  public TFLayerBase getTfLayer() {
    if (tfLayer == null) return null;
    return (TFLayerBase) tfLayer.copy();
  }

  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void graphModel() {
    getLog().eval(() -> {
      TFLayerBase tfLayer1 = getTfLayer();
      GraphDef graphDef = tfLayer1.constGraph();
      tfLayer1.freeRef();
      GraphModel graphModel = new GraphModel(graphDef.toByteArray());
      return JsonUtil.toJson(graphModel);
    });
  }

  @AfterEach
  void cleanup() {
    if (null != tfLayer) {
      tfLayer.freeRef();
    }
  }

  protected abstract @Nonnull
  TFLayerBase createTFLayer();

  @Nonnull
  @Override
  protected Layer lossLayer() {
    return new MeanSqLossLayer();
  }
}
