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

import com.simiacryptus.ref.lang.RefUtil;
import org.junit.After;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

public class LRNLayerTest extends RawTFLayerTestBase {

  private final TFLayerBase tfLayer = createTFLayer();

  public LRNLayerTest() {
    validateDifferentials = false;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{2, 3, 20}};
  }

  @After
  public void cleanup() {
    super.cleanup();
    if (null != tfLayer)
      tfLayer.freeRef();
  }

  @Nonnull
  protected TFLayerBase createTFLayer() {
    LRNLayer temp_21_0002 = new LRNLayer();
    temp_21_0002.setRadius(5);
    LRNLayer temp_21_0003 = temp_21_0002.addRef();
    temp_21_0003.setAlpha(1e-4f);
    LRNLayer temp_21_0004 = temp_21_0003.addRef();
    temp_21_0004.setBias((float) 2);
    LRNLayer temp_21_0001 = temp_21_0004.addRef();
    temp_21_0004.freeRef();
    temp_21_0003.freeRef();
    temp_21_0002.freeRef();
    return temp_21_0001;
  }

}
