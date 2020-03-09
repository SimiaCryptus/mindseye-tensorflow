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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.util.TFConverter;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.tensorflow.GraphModel;

import javax.annotation.Nonnull;
import java.util.Random;

public abstract class Conv2DLayerTest extends RawTFLayerTestBase {

  public Conv2DLayerTest() {
    validateDifferentials = false;
    testTraining = false;
  }

  @Nonnull
  public TFLayerBase createTFLayer() {
    Conv2DLayer layer = new Conv2DLayer(3, 3, 1, 1);
    layer.setStrideX(2);
    layer.setStrideY(2);
    RefMap<String, Tensor> weights = layer.getWeights();
    assert weights != null;
    Tensor kernel = weights.get("kernel");
    weights.freeRef();
    assert kernel != null;
    kernel.randomize(1.0);
    kernel.freeRef();
    TFLayer constLayer = layer.asConstLayer();
    layer.freeRef();
    //    kernel.set(new int[]{1,1,0}, 1.0);
    //    kernel.set(new int[]{0,0,0}, 1.0);
    return constLayer;
  }

  @Nonnull
  @Override
  public abstract int[][] getSmallDims(Random random);

  public static class Small_0 extends Conv2DLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{6, 6, 1}};
    }
  }

  public static class Small_1 extends Conv2DLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{5, 5, 1}};
    }

  }

  public static class Direct_1 extends Conv2DLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{5, 5, 1}};
    }

    @Nonnull
    @Override
    public TFLayerBase createTFLayer() {
      Conv2DLayer layer = new Conv2DLayer(3, 3, 1, 1);
      layer.setStrideX(2);
      layer.setStrideY(2);
      RefMap<String, Tensor> weights = layer.getWeights();
      assert weights != null;
      Tensor kernel = weights.get("kernel");
      weights.freeRef();
      assert kernel != null;
      kernel.randomize(1.0);
      kernel.freeRef();
      //    kernel.set(new int[]{0,0,0}, 1.0);
      return layer;
    }

  }

  public static class Multiband_0 extends Conv2DLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{5, 5, 1}};
    }

    @Nonnull
    public TFLayerBase createTFLayer() {
      Conv2DLayer layer = new Conv2DLayer(3, 3, 1, 2);
      layer.setStrideX(1);
      layer.setStrideY(1);
      RefMap<String, Tensor> weights = layer.getWeights();
      assert weights != null;
      Tensor kernel = weights.get("kernel");
      weights.freeRef();
      assert kernel != null;
      kernel.randomize(1.0);
      kernel.freeRef();
      TFLayer constLayer = layer.asConstLayer();
      layer.freeRef();
      return constLayer;
    }

  }

  public static class Multiband_1 extends Conv2DLayerTest {

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{5, 5, 2}};
    }

    @Nonnull
    public TFLayerBase createTFLayer() {
      Conv2DLayer layer = new Conv2DLayer(3, 3, 2, 2);
      layer.setStrideX(1);
      layer.setStrideY(1);
      RefMap<String, Tensor> weights = layer.getWeights();
      assert weights != null;
      Tensor kernel = weights.get("kernel");
      weights.freeRef();
      assert kernel != null;
      kernel.randomize(1.0);
      kernel.freeRef();
      TFLayer constLayer = layer.asConstLayer();
      layer.freeRef();
      return constLayer;
    }
  }

  public static class Img_0 extends Conv2DLayerTest {
    protected final int padding = 3;

    @Override
    public Layer getReferenceLayer() {
      return new TFConverter() {
        @Nonnull
        @Override
        protected Layer getConv2D(@Nonnull GraphModel.GraphNode graphNode) {
          SimpleConvolutionLayer convolutionLayer = (SimpleConvolutionLayer) super.getConv2D(graphNode);
          convolutionLayer.setPaddingX(padding);
          convolutionLayer.setPaddingY(padding);
          return convolutionLayer;
        }
      }.convert(getTfLayer());
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{320, 240, 3}};
    }

    @Nonnull
    public TFLayerBase createTFLayer() {
      Conv2DLayer layer = new Conv2DLayer(7, 7, 3, 64);
      layer.setStrideX(2);
      layer.setStrideY(2);
      RefMap<String, Tensor> weights = layer.getWeights();
      assert weights != null;
      Tensor kernel = weights.get("kernel");
      weights.freeRef();
      assert kernel != null;
      kernel.randomize(1.0);
      kernel.freeRef();
      TFLayer constLayer = layer.asConstLayer();
      layer.freeRef();
      return constLayer;
      //      return tfLayer;
    }

  }

  public static class Stride3_0 extends Conv2DLayerTest {

    private final int padding;

    public Stride3_0() {
      this(1);
    }

    protected Stride3_0(int padding) {
      this.padding = padding;
    }

    @Override
    public Layer getReferenceLayer() {
      return new TFConverter() {
        @Nonnull
        @Override
        protected Layer getConv2D(@Nonnull GraphModel.GraphNode graphNode) {
          SimpleConvolutionLayer convolutionLayer = (SimpleConvolutionLayer) super.getConv2D(graphNode);
          convolutionLayer.setPaddingX(padding);
          convolutionLayer.setPaddingY(padding);
          return convolutionLayer;
        }
      }.convert(getTfLayer());
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{8, 8, 1}};
    }

    @Nonnull
    public TFLayerBase createTFLayer() {
      Conv2DLayer layer = new Conv2DLayer(3, 3, 1, 1);
      layer.setStrideX(3);
      layer.setStrideY(3);
      RefMap<String, Tensor> weights = layer.getWeights();
      assert weights != null;
      Tensor kernel = weights.get("kernel");
      weights.freeRef();
      assert kernel != null;
      kernel.randomize(1.0);
      kernel.freeRef();
      TFLayer constLayer = layer.asConstLayer();
      layer.freeRef();
      //      kernel.set(new int[]{0,0,0}, 1.0);
      return constLayer;
      //return tfLayer;
    }

  }

  public static class Stride3_1 extends Stride3_0 {

    public Stride3_1() {
      super(0);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{9, 9, 1}};
    }

  }

  public static class Stride4_0 extends Conv2DLayerTest {

    private final int padding;

    public Stride4_0() {
      this(1);
    }

    protected Stride4_0(int padding) {
      this.padding = padding;
    }

    @Override
    public Layer getReferenceLayer() {
      return new TFConverter() {
        @Nonnull
        @Override
        protected Layer getConv2D(@Nonnull GraphModel.GraphNode graphNode) {
          SimpleConvolutionLayer convolutionLayer = (SimpleConvolutionLayer) super.getConv2D(graphNode);
          convolutionLayer.setPaddingX(padding);
          convolutionLayer.setPaddingY(padding);
          return convolutionLayer;
        }
      }.convert(getTfLayer());
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{8, 8, 1}};
    }

    @Nonnull
    public TFLayerBase createTFLayer() {
      Conv2DLayer layer = new Conv2DLayer(3, 3, 1, 1);
      layer.setStrideX(4);
      layer.setStrideY(4);
      RefMap<String, Tensor> weights = layer.getWeights();
      assert weights != null;
      Tensor kernel = weights.get("kernel");
      weights.freeRef();
      assert kernel != null;
      kernel.randomize(1.0);
      kernel.freeRef();
      TFLayer constLayer = layer.asConstLayer();
      layer.freeRef();
      //      kernel.set(new int[]{0,0,0}, 1.0);
      return constLayer;
      //return tfLayer;
    }

  }

  public static class Stride4_1 extends Stride4_0 {

    public Stride4_1() {
      super(1);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{10, 10, 1}};
    }

  }

  public static class Stride4_2 extends Stride4_0 {

    public Stride4_2() {
      super(0);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{11, 11, 1}};
    }

  }

  public static class Stride4_3 extends Stride4_0 {

    public Stride4_3() {
      super(1);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{9, 9, 1}};
    }

  }

}
