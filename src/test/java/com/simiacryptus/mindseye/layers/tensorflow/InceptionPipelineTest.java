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
import com.simiacryptus.mindseye.util.TFConverter;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.tensorflow.GraphModel;
import com.simiacryptus.tensorflow.ImageNetworkPipeline;
import com.simiacryptus.util.JsonUtil;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public abstract class InceptionPipelineTest extends LayerTestBase {
  private static final List<TFLayer> layers = getLayers();

  public InceptionPipelineTest() {
    validateDifferentials = false;
    testTraining = false;
    this.testingBatchSize = 5;
  }

  public static List<TFLayer> getLayers() {
    final ImageNetworkPipeline inception5h = ImageNetworkPipeline.inception5h();
    return IntStream.range(0, inception5h.graphDefs.size()).mapToObj(i -> new TFLayer(
        inception5h.graphDefs.get(i).toByteArray(),
        new HashMap<>(),
        inception5h.nodeIds().get(i),
        i == 0 ? "input" : inception5h.nodeIds().get(i - 1)
    ).setFloat(true)).collect(Collectors.toList());
  }

  @Override
  public void run(@Nonnull NotebookOutput log) {
    log.eval(() -> {
      TFLayer tfLayer = tfLayer();
      GraphDef graphDef = tfLayer.constGraph();
      GraphModel graphModel = new GraphModel(graphDef.toByteArray());
      return JsonUtil.toJson(graphModel);
    });
    super.run(log);
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    TFLayer tfLayer = tfLayer();
    return new TFConverter().convert(tfLayer);
  }

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    return tfLayer();
  }

  @NotNull
  public abstract TFLayer tfLayer();

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  @Override
  protected Layer lossLayer() {
    return new MeanSqLossLayer();
  }

  public static class Layer0 extends InceptionPipelineTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
          {320, 240, 3}
      };
    }


    @NotNull
    public TFLayer tfLayer() {
      return (TFLayer) layers.get(0).copy();
    }

  }

  public static class Layer1 extends InceptionPipelineTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
          {160, 120, 64}
      };
    }

    @NotNull
    public TFLayer tfLayer() {
      return (TFLayer) layers.get(1).copy();
    }


  }

  public static class Layer2 extends InceptionPipelineTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
          {80, 60, 192}
      };
    }

    @NotNull
    public TFLayer tfLayer() {
      return (TFLayer) layers.get(2).copy();
    }

  }


}
