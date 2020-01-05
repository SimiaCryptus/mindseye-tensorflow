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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.tensorflow.GraphModel;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Random;

public abstract @RefAware
class Conv2DLayerTest extends RawTFLayerTestBase {

  public Conv2DLayerTest() {
    validateDifferentials = false;
    testTraining = false;
  }

  public static @SuppressWarnings("unused")
  Conv2DLayerTest[] addRefs(Conv2DLayerTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(Conv2DLayerTest::addRef)
        .toArray((x) -> new Conv2DLayerTest[x]);
  }

  public static @SuppressWarnings("unused")
  Conv2DLayerTest[][] addRefs(Conv2DLayerTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(Conv2DLayerTest::addRefs)
        .toArray((x) -> new Conv2DLayerTest[x][]);
  }

  @NotNull
  public TFLayerBase createTFLayer() {
    Conv2DLayer layer = new Conv2DLayer(3, 3, 1, 1).setStrideX(2).setStrideY(2);
    Tensor kernel = layer.getWeights().get("kernel");
    kernel.randomize(1.0);
    //    kernel.set(new int[]{1,1,0}, 1.0);
    //    kernel.set(new int[]{0,0,0}, 1.0);
    return layer.asConstLayer();
  }

  @Nonnull
  @Override
  public abstract int[][] getSmallDims(Random random);

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  Conv2DLayerTest addRef() {
    return (Conv2DLayerTest) super.addRef();
  }

  public static @RefAware
  class Small_0 extends Conv2DLayerTest {
    public static @SuppressWarnings("unused")
    Small_0[] addRefs(Small_0[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Small_0::addRef)
          .toArray((x) -> new Small_0[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{6, 6, 1}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Small_0 addRef() {
      return (Small_0) super.addRef();
    }
  }

  public static @RefAware
  class Small_1 extends Conv2DLayerTest {
    public static @SuppressWarnings("unused")
    Small_1[] addRefs(Small_1[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Small_1::addRef)
          .toArray((x) -> new Small_1[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{5, 5, 1}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Small_1 addRef() {
      return (Small_1) super.addRef();
    }
  }

  public static @RefAware
  class Direct_1 extends Conv2DLayerTest {
    public static @SuppressWarnings("unused")
    Direct_1[] addRefs(Direct_1[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Direct_1::addRef)
          .toArray((x) -> new Direct_1[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{5, 5, 1}};
    }

    @NotNull
    @Override
    public TFLayerBase createTFLayer() {
      Conv2DLayer layer = new Conv2DLayer(3, 3, 1, 1).setStrideX(2).setStrideY(2);
      Tensor kernel = layer.getWeights().get("kernel");
      kernel.randomize(1.0);
      //    kernel.set(new int[]{0,0,0}, 1.0);
      return layer;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Direct_1 addRef() {
      return (Direct_1) super.addRef();
    }
  }

  public static @RefAware
  class Multiband_0 extends Conv2DLayerTest {

    public static @SuppressWarnings("unused")
    Multiband_0[] addRefs(Multiband_0[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Multiband_0::addRef)
          .toArray((x) -> new Multiband_0[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{5, 5, 1}};
    }

    @NotNull
    public TFLayerBase createTFLayer() {
      Conv2DLayer layer = new Conv2DLayer(3, 3, 1, 2).setStrideX(1).setStrideY(1);
      Tensor kernel = layer.getWeights().get("kernel");
      kernel.randomize(1.0);
      return layer.asConstLayer();
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Multiband_0 addRef() {
      return (Multiband_0) super.addRef();
    }

  }

  public static @RefAware
  class Multiband_1 extends Conv2DLayerTest {

    public static @SuppressWarnings("unused")
    Multiband_1[] addRefs(Multiband_1[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Multiband_1::addRef)
          .toArray((x) -> new Multiband_1[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{5, 5, 2}};
    }

    @NotNull
    public TFLayerBase createTFLayer() {
      Conv2DLayer layer = new Conv2DLayer(3, 3, 2, 2).setStrideX(1).setStrideY(1);
      Tensor kernel = layer.getWeights().get("kernel");
      kernel.randomize(1.0);
      return layer.asConstLayer();
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Multiband_1 addRef() {
      return (Multiband_1) super.addRef();
    }

  }

  public static @RefAware
  class Img_0 extends Conv2DLayerTest {
    protected final int padding = 3;

    @Override
    public Layer getReferenceLayer() {
      return new TFConverter() {
        @Override
        protected Layer getConv2D(GraphModel.GraphNode graphNode) {
          SimpleConvolutionLayer convolutionLayer = (SimpleConvolutionLayer) super.getConv2D(graphNode);
          convolutionLayer.setPaddingX(padding);
          convolutionLayer.setPaddingY(padding);
          return convolutionLayer;
        }
      }.convert(getTfLayer());
    }

    public static @SuppressWarnings("unused")
    Img_0[] addRefs(Img_0[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Img_0::addRef).toArray((x) -> new Img_0[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{320, 240, 3}};
    }

    @NotNull
    public TFLayerBase createTFLayer() {
      Conv2DLayer layer = new Conv2DLayer(7, 7, 3, 64).setStrideX(2).setStrideY(2);
      Tensor kernel = layer.getWeights().get("kernel");
      kernel.randomize(1.0);
      return layer.asConstLayer();
      //      return tfLayer;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Img_0 addRef() {
      return (Img_0) super.addRef();
    }
  }

  public static @RefAware
  class Stride3_0 extends Conv2DLayerTest {

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
        @Override
        protected Layer getConv2D(GraphModel.GraphNode graphNode) {
          SimpleConvolutionLayer convolutionLayer = (SimpleConvolutionLayer) super.getConv2D(graphNode);
          convolutionLayer.setPaddingX(padding);
          convolutionLayer.setPaddingY(padding);
          return convolutionLayer;
        }
      }.convert(getTfLayer());
    }

    public static @SuppressWarnings("unused")
    Stride3_0[] addRefs(Stride3_0[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Stride3_0::addRef)
          .toArray((x) -> new Stride3_0[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{8, 8, 1}};
    }

    @NotNull
    public TFLayerBase createTFLayer() {
      Conv2DLayer layer = new Conv2DLayer(3, 3, 1, 1).setStrideX(3).setStrideY(3);
      Tensor kernel = layer.getWeights().get("kernel");
      kernel.randomize(1.0);
      //      kernel.set(new int[]{0,0,0}, 1.0);
      return layer.asConstLayer();
      //return tfLayer;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Stride3_0 addRef() {
      return (Stride3_0) super.addRef();
    }
  }

  public static @RefAware
  class Stride3_1 extends Stride3_0 {

    public Stride3_1() {
      super(0);
    }

    public static @SuppressWarnings("unused")
    Stride3_1[] addRefs(Stride3_1[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Stride3_1::addRef)
          .toArray((x) -> new Stride3_1[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{9, 9, 1}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Stride3_1 addRef() {
      return (Stride3_1) super.addRef();
    }

  }

  public static @RefAware
  class Stride4_0 extends Conv2DLayerTest {

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
        @Override
        protected Layer getConv2D(GraphModel.GraphNode graphNode) {
          SimpleConvolutionLayer convolutionLayer = (SimpleConvolutionLayer) super.getConv2D(graphNode);
          convolutionLayer.setPaddingX(padding);
          convolutionLayer.setPaddingY(padding);
          return convolutionLayer;
        }
      }.convert(getTfLayer());
    }

    public static @SuppressWarnings("unused")
    Stride4_0[] addRefs(Stride4_0[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Stride4_0::addRef)
          .toArray((x) -> new Stride4_0[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{8, 8, 1}};
    }

    @NotNull
    public TFLayerBase createTFLayer() {
      Conv2DLayer layer = new Conv2DLayer(3, 3, 1, 1).setStrideX(4).setStrideY(4);
      Tensor kernel = layer.getWeights().get("kernel");
      kernel.randomize(1.0);
      //      kernel.set(new int[]{0,0,0}, 1.0);
      return layer.asConstLayer();
      //return tfLayer;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Stride4_0 addRef() {
      return (Stride4_0) super.addRef();
    }
  }

  public static @RefAware
  class Stride4_1 extends Stride4_0 {

    public Stride4_1() {
      super(1);
    }

    public static @SuppressWarnings("unused")
    Stride4_1[] addRefs(Stride4_1[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Stride4_1::addRef)
          .toArray((x) -> new Stride4_1[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{10, 10, 1}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Stride4_1 addRef() {
      return (Stride4_1) super.addRef();
    }

  }

  public static @RefAware
  class Stride4_2 extends Stride4_0 {

    public Stride4_2() {
      super(0);
    }

    public static @SuppressWarnings("unused")
    Stride4_2[] addRefs(Stride4_2[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Stride4_2::addRef)
          .toArray((x) -> new Stride4_2[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{11, 11, 1}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Stride4_2 addRef() {
      return (Stride4_2) super.addRef();
    }

  }

  public static @RefAware
  class Stride4_3 extends Stride4_0 {

    public Stride4_3() {
      super(1);
    }

    public static @SuppressWarnings("unused")
    Stride4_3[] addRefs(Stride4_3[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Stride4_3::addRef)
          .toArray((x) -> new Stride4_3[x]);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{{9, 9, 1}};
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    Stride4_3 addRef() {
      return (Stride4_3) super.addRef();
    }

  }

}
