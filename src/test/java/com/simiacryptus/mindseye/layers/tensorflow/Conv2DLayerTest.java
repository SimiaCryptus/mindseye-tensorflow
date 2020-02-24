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
    Conv2DLayer temp_07_0007 = new Conv2DLayer(3, 3, 1, 1);
    temp_07_0007.setStrideX(2);
    Conv2DLayer temp_07_0014 = temp_07_0007.addRef();
    temp_07_0014.setStrideY(2);
    Conv2DLayer layer = temp_07_0014.addRef();
    temp_07_0014.freeRef();
    temp_07_0007.freeRef();
    RefMap<String, Tensor> temp_07_0015 = layer.getWeights();
    assert temp_07_0015 != null;
    Tensor kernel = temp_07_0015.get("kernel");
    temp_07_0015.freeRef();
    assert kernel != null;
    kernel.randomize(1.0);
    kernel.freeRef();
    TFLayer temp_07_0001 = layer.asConstLayer();
    layer.freeRef();
    //    kernel.set(new int[]{1,1,0}, 1.0);
    //    kernel.set(new int[]{0,0,0}, 1.0);
    return temp_07_0001;
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
      Conv2DLayer temp_07_0008 = new Conv2DLayer(3, 3, 1, 1);
      temp_07_0008.setStrideX(2);
      Conv2DLayer temp_07_0016 = temp_07_0008.addRef();
      temp_07_0016.setStrideY(2);
      Conv2DLayer layer = temp_07_0016.addRef();
      temp_07_0016.freeRef();
      temp_07_0008.freeRef();
      RefMap<String, Tensor> temp_07_0017 = layer.getWeights();
      assert temp_07_0017 != null;
      Tensor kernel = temp_07_0017.get("kernel");
      temp_07_0017.freeRef();
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
      Conv2DLayer temp_07_0009 = new Conv2DLayer(3, 3, 1, 2);
      temp_07_0009.setStrideX(1);
      Conv2DLayer temp_07_0018 = temp_07_0009.addRef();
      temp_07_0018.setStrideY(1);
      Conv2DLayer layer = temp_07_0018.addRef();
      temp_07_0018.freeRef();
      temp_07_0009.freeRef();
      RefMap<String, Tensor> temp_07_0019 = layer.getWeights();
      assert temp_07_0019 != null;
      Tensor kernel = temp_07_0019.get("kernel");
      temp_07_0019.freeRef();
      assert kernel != null;
      kernel.randomize(1.0);
      kernel.freeRef();
      TFLayer temp_07_0002 = layer.asConstLayer();
      layer.freeRef();
      return temp_07_0002;
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
      Conv2DLayer temp_07_0010 = new Conv2DLayer(3, 3, 2, 2);
      temp_07_0010.setStrideX(1);
      Conv2DLayer temp_07_0020 = temp_07_0010.addRef();
      temp_07_0020.setStrideY(1);
      Conv2DLayer layer = temp_07_0020.addRef();
      temp_07_0020.freeRef();
      temp_07_0010.freeRef();
      RefMap<String, Tensor> temp_07_0021 = layer.getWeights();
      assert temp_07_0021 != null;
      Tensor kernel = temp_07_0021.get("kernel");
      temp_07_0021.freeRef();
      assert kernel != null;
      kernel.randomize(1.0);
      kernel.freeRef();
      TFLayer temp_07_0003 = layer.asConstLayer();
      layer.freeRef();
      return temp_07_0003;
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
      Conv2DLayer temp_07_0011 = new Conv2DLayer(7, 7, 3, 64);
      temp_07_0011.setStrideX(2);
      Conv2DLayer temp_07_0022 = temp_07_0011.addRef();
      temp_07_0022.setStrideY(2);
      Conv2DLayer layer = temp_07_0022.addRef();
      temp_07_0022.freeRef();
      temp_07_0011.freeRef();
      RefMap<String, Tensor> temp_07_0023 = layer.getWeights();
      assert temp_07_0023 != null;
      Tensor kernel = temp_07_0023.get("kernel");
      temp_07_0023.freeRef();
      assert kernel != null;
      kernel.randomize(1.0);
      kernel.freeRef();
      TFLayer temp_07_0004 = layer.asConstLayer();
      layer.freeRef();
      return temp_07_0004;
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
      Conv2DLayer temp_07_0012 = new Conv2DLayer(3, 3, 1, 1);
      temp_07_0012.setStrideX(3);
      Conv2DLayer temp_07_0024 = temp_07_0012.addRef();
      temp_07_0024.setStrideY(3);
      Conv2DLayer layer = temp_07_0024.addRef();
      temp_07_0024.freeRef();
      temp_07_0012.freeRef();
      RefMap<String, Tensor> temp_07_0025 = layer.getWeights();
      assert temp_07_0025 != null;
      Tensor kernel = temp_07_0025.get("kernel");
      temp_07_0025.freeRef();
      assert kernel != null;
      kernel.randomize(1.0);
      kernel.freeRef();
      TFLayer temp_07_0005 = layer.asConstLayer();
      layer.freeRef();
      //      kernel.set(new int[]{0,0,0}, 1.0);
      return temp_07_0005;
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
      Conv2DLayer temp_07_0013 = new Conv2DLayer(3, 3, 1, 1);
      temp_07_0013.setStrideX(4);
      Conv2DLayer temp_07_0026 = temp_07_0013.addRef();
      temp_07_0026.setStrideY(4);
      Conv2DLayer layer = temp_07_0026.addRef();
      temp_07_0026.freeRef();
      temp_07_0013.freeRef();
      RefMap<String, Tensor> temp_07_0027 = layer.getWeights();
      assert temp_07_0027 != null;
      Tensor kernel = temp_07_0027.get("kernel");
      temp_07_0027.freeRef();
      assert kernel != null;
      kernel.randomize(1.0);
      kernel.freeRef();
      TFLayer temp_07_0006 = layer.asConstLayer();
      layer.freeRef();
      //      kernel.set(new int[]{0,0,0}, 1.0);
      return temp_07_0006;
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
