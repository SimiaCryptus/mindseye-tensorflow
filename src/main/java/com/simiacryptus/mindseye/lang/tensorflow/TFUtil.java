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

package com.simiacryptus.mindseye.lang.tensorflow;

import com.simiacryptus.lang.UncheckedConsumer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.tensorflow.TensorflowUtil;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.framework.*;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;


public class TFUtil {

  public static void launchTensorboard(File logDir, UncheckedConsumer<Process> waiter) throws IOException, URISyntaxException {
    Process tensorboard = new ProcessBuilder().command(
        System.getProperty("tensorboard", System.getProperty("user.home") + "\\AppData\\Local\\Programs\\Python\\Python36\\Scripts\\tensorboard.exe"),
        "--logdir=" + logDir.getAbsolutePath()
    ).start();
    Desktop.getDesktop().browse(new URI("http://localhost:6006/"));
    try {
      try {
        waiter.accept(tensorboard);
      } catch (RuntimeException e) {
        throw e;
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    } finally {
      if (tensorboard.isAlive()) tensorboard.destroyForcibly();
    }
  }

  @NotNull
  public static GraphDef implantConstants(GraphDef graphDef, Map<String, Tensor> weights) {
    graphDef = TensorflowUtil.editGraph(graphDef, graphBuilder -> {
      weights.forEach((key, value) -> {
        TensorflowUtil.editNode(graphBuilder, key, (NodeDef.Builder node) -> {
              DataType type = node.getAttrMap().get("dtype").getType();
              TensorProto.Builder tensor = TensorProto.newBuilder();
              if (type == DataType.DT_DOUBLE) {
                tensor.addAllDoubleVal(Arrays.stream(value.getData()).mapToObj(x -> x).collect(Collectors.toList()));
              } else if (type == DataType.DT_FLOAT) {
                tensor.addAllFloatVal(Arrays.stream(value.getData()).mapToObj(x -> (float)x).collect(Collectors.toList()));
              } else {
                throw new UnsupportedOperationException(type.toString());
              }
              tensor.setDtype(type);
              tensor.setTensorShape(node.getAttrMap().get("shape").getShape());
              return node
                  .removeAttr("shape")
                  .putAttr("value", AttrValue.newBuilder()
                      .setTensor(tensor.build())
                      .build())
                  .setOp("Const");
            }
        );
      });
      return graphBuilder;
    });
    return graphDef;
  }

}
