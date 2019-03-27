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
import com.simiacryptus.mindseye.util.TFConverter;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.tensorflow.GraphModel;
import com.simiacryptus.util.JsonUtil;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;


public abstract class MaxPoolLayerTest extends RawTFLayerTestBase {


  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
        {5, 5, 1}
    };
  }


  public static class Test0 extends MaxPoolLayerTest {
    @NotNull
    protected TFLayerBase createTFLayer() {
      MaxPoolLayer maxPoolLayer = new MaxPoolLayer();
      maxPoolLayer.setWidth(2);
      maxPoolLayer.setHeight(2);
      maxPoolLayer.setStrideX(1);
      maxPoolLayer.setStrideY(1);
      return maxPoolLayer;
    }

  }

  public static class Test1 extends MaxPoolLayerTest {
    @NotNull
    protected TFLayerBase createTFLayer() {
      MaxPoolLayer maxPoolLayer = new MaxPoolLayer();
      maxPoolLayer.setWidth(2);
      maxPoolLayer.setHeight(2);
      maxPoolLayer.setStrideX(2);
      maxPoolLayer.setStrideY(2);
      return maxPoolLayer;
    }

  }

  public static class Test2 extends MaxPoolLayerTest {
    @NotNull
    protected TFLayerBase createTFLayer() {
      MaxPoolLayer maxPoolLayer = new MaxPoolLayer();
      maxPoolLayer.setWidth(3);
      maxPoolLayer.setHeight(3);
      maxPoolLayer.setStrideX(2);
      maxPoolLayer.setStrideY(2);
      return maxPoolLayer;
    }

  }

}
