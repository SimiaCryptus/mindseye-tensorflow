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

package com.simiacryptus.mindseye.examples.mnist;

import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.tensorflow.TFUtil;
import com.simiacryptus.mindseye.layers.StochasticComponent;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.tensorflow.TFLayerBase;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.test.data.MNIST;
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.tensorflow.TensorboardEventWriter;
import com.simiacryptus.util.CodeUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.util.test.TestSettings;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.framework.GraphDef;
import smile.plot.swing.PlotCanvas;
import smile.plot.swing.ScatterPlot;

import javax.annotation.Nonnull;
import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Comparator;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public abstract class MnistDemoBase {
  private static final Logger log = LoggerFactory.getLogger(MnistDemoBase.class);
  protected int timeout = 60;

  protected abstract byte[] getGraphDef();

  @Test
  public void test() throws Exception {
    File tensorboardLocation = new File("tensorboard/tflayer_TFLayer").getAbsoluteFile();
    File[] listFiles = tensorboardLocation.getParentFile().listFiles();
    if (null != listFiles)
      RefArrays.stream(listFiles).forEach(file -> {
        System.out.println("Delete: " + file);
        file.delete();
      });
    byte[] graphDef = getGraphDef();
    if (null != graphDef) {
      TFLayerBase.eventWriter = new TensorboardEventWriter(tensorboardLocation, GraphDef.parseFrom(graphDef));
    }
    File reportFile = new File(RefString.format("target/reports/%s/%s/test", getClass().getSimpleName(),
        new SimpleDateFormat("yyyyMMddHHmm").format(new Date())));
    MarkdownNotebookOutput log = new MarkdownNotebookOutput(reportFile, true);
    try (CodeUtil.LogInterception ignored = CodeUtil.intercept(log, ReferenceCountingBase.class.getCanonicalName())) {
      run(log);
    } finally {
      log.close();
    }
    if (null != TFLayerBase.eventWriter) {
      TFLayerBase.eventWriter.close();
      TFLayerBase.eventWriter = null;
    }
    if (null != graphDef && TestSettings.INSTANCE.isInteractive) {
      TFUtil.launchTensorboard(tensorboardLocation.getParentFile(), x -> {
        JOptionPane.showConfirmDialog(null, "Press OK to exit");
      });
    }
  }

  public void run(@Nonnull NotebookOutput log) {

    final Tensor[][] trainingData = MNIST.trainingDataStream().map(labeledObject1 -> {
      @Nonnull final Tensor categoryTensor = new Tensor(10);
      final int category = parse(labeledObject1.label);
      categoryTensor.set(category, 1);
      Tensor data = labeledObject1.data.addRef();
      labeledObject1.freeRef();
      return new Tensor[]{data, categoryTensor};
    }).toArray(i1 -> new Tensor[i1][]);

    log.h1("Model");
    final Layer recognitionNetwork = buildModel(log);

    log.h1("Training");
    @Nonnull final RefList<Step> history = new RefArrayList<>();
    @Nonnull final TrainingMonitor monitor = new TrainingMonitor() {
      @Override
      public void clear() {
        super.clear();
      }

      @Override
      public void log(final String msg) {
        MnistDemoBase.log.info(msg);
        //        if(null != TFLayerBase.eventWriter) {
        //          try {
        //            TFLayerBase.eventWriter.write(LogMessage.newBuilder().setLevel(LogMessage.Level.INFO).setMessage(msg).build());
        //          } catch (IOException e) {
        //            throw new RuntimeException(e);
        //          }
        //        }
        super.log(msg);
      }

      @Override
      public void onStepComplete(@Nonnull final Step currentPoint) {
        if (null != TFLayerBase.eventWriter) {
          TFLayerBase.eventWriter.setStep(currentPoint.iteration);
        }
        history.add(currentPoint.addRef());
        super.onStepComplete(currentPoint);
      }
    };
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<Double>) () -> {
      EntropyLossLayer loss = new EntropyLossLayer();
      @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(
          recognitionNetwork == null ? null : recognitionNetwork.addRef(), loss.addRef());
      loss.freeRef();
      @Nonnull final Trainable trainable = new SampledArrayTrainable(RefUtil.addRef(trainingData),
          supervisedNetwork, 1000, 1000);
      IterativeTrainer temp_06_0008 = new IterativeTrainer(trainable);
      temp_06_0008.setMonitor(monitor);
      IterativeTrainer temp_06_0009 = temp_06_0008.addRef();
      temp_06_0009.setOrientation(new LBFGS());
      IterativeTrainer temp_06_0010 = temp_06_0009.addRef();
      temp_06_0010.setLineSearchFactory(n -> new QuadraticSearch());
      //          .setLineSearchFactory(n -> new ArmijoWolfeSearch().setAlpha(1e0))
      IterativeTrainer temp_06_0011 = temp_06_0010.addRef();
      temp_06_0011.setTimeout(timeout, TimeUnit.SECONDS);
      IterativeTrainer temp_06_0012 = temp_06_0011.addRef();
      temp_06_0012.setMaxIterations(200);
      IterativeTrainer temp_06_0013 = temp_06_0012.addRef();
      temp_06_0013.setIterationsPerSample(20);
      IterativeTrainer temp_06_0014 = temp_06_0013.addRef();
      double temp_06_0002 = temp_06_0014.run();
      temp_06_0014.freeRef();
      temp_06_0013.freeRef();
      temp_06_0012.freeRef();
      temp_06_0011.freeRef();
      temp_06_0010.freeRef();
      temp_06_0009.freeRef();
      temp_06_0008.freeRef();
      //          .setLineSearchFactory(n -> new ArmijoWolfeSearch().setAlpha(1e0))
      return temp_06_0002;
    }, RefUtil.addRef(trainingData), recognitionNetwork == null ? null : recognitionNetwork.addRef()));
    RefUtil.freeRef(trainingData);
    if (!history.isEmpty()) {
      log.eval(RefUtil.wrapInterface((UncheckedSupplier<PlotCanvas>) () -> {
        @Nonnull final PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> {
          assert step.point != null;
          double[] temp_06_0003 = new double[]{step.iteration, Math.log10(step.point.getMean())};
          step.freeRef();
          return temp_06_0003;
        }).toArray(i -> new double[i][]));
        plot.setTitle("Convergence Plot");
        plot.setAxisLabels("Iteration", "log10(Fitness)");
        plot.setSize(600, 400);
        return plot;
      }, history.addRef()));
    }

    history.freeRef();
    if (recognitionNetwork instanceof DAGNetwork) {
      ((DAGNetwork) recognitionNetwork).visitLayers(layer -> {
        if (layer instanceof StochasticComponent)
          ((StochasticComponent) layer).clearNoise();
        if (null != layer)
          layer.freeRef();
      });
    }

    @Nonnull final String modelName = "model.json";
    assert recognitionNetwork != null;
    log.p("Saved model as " + log.file(recognitionNetwork.getJson().toString(), modelName, modelName));

    if (null != TFLayerBase.eventWriter) {
      try {
        TFLayerBase.eventWriter.close();
      } catch (IOException e) {
        throw Util.throwException(e);
      }
      TFLayerBase.eventWriter = null;
    }

    log.h1("Validation");
    log.p("If we apply our model against the entire validation dataset, we get this accuracy:");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<Double>) () -> {
      RefList<LabeledObject<Tensor>> validation = MNIST.validationDataStream().collect(RefCollectors.toList());
      Tensor[][] tensors = new Tensor[][]{validation.stream().map(labeledObject -> {
        Tensor data = labeledObject.data.addRef();
        labeledObject.freeRef();
        return data;
      }).toArray(i -> new Tensor[i])};
      Result temp_06_0015 = recognitionNetwork.eval(RefUtil.addRef(tensors));
      assert temp_06_0015 != null;
      TensorList predictionData = Result.getData(temp_06_0015);
      RefUtil.freeRef(tensors);
      int length = predictionData.length();
      List<int[]> predicitonList = IntStream.range(0, length)
          .mapToObj(rowIndex -> {
            Tensor predictionTensor = predictionData.get(rowIndex);
            int[] ints = RefIntStream.range(0, 10).mapToObj(x -> x).sorted(
                Comparator.comparingDouble(ii -> -predictionTensor.get(ii))
            ).mapToInt(x -> x).toArray();
            predictionTensor.freeRef();
            return ints;
          }).collect(Collectors.toList());
      predictionData.freeRef();
      return RefIntStream.range(0, length)
          .mapToDouble(RefUtil.wrapInterface(rowIndex -> {
            LabeledObject<Tensor> labeledObject = validation.get(rowIndex);
            int i = predicitonList.get(rowIndex)[0] == parse(labeledObject.label) ? 1 : 0;
            labeledObject.freeRef();
            return i;
          }, validation))
          .average().getAsDouble() * 100;
    }, recognitionNetwork.addRef()));

    log.p("Let's examine some incorrectly predicted results in more detail:");
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<TableOutput>) () -> {
      @Nonnull final TableOutput table = new TableOutput();
      MNIST.validationDataStream().map(RefUtil.wrapInterface(
          (Function<? super LabeledObject<Tensor>, LinkedHashMap<CharSequence, Object>>) labeledObject -> {
            final int actualCategory = parse(labeledObject.label);
            Result result = recognitionNetwork.eval(labeledObject.data.addRef());
            assert result != null;
            TensorList tensorList = result.getData();
            Tensor tensor = tensorList.get(0);
            tensorList.freeRef();
            result.freeRef();
            final int[] predictionList = RefIntStream.range(0, 10).mapToObj(x -> x)
                .sorted(RefComparator.comparingDouble(i -> -tensor.get(i))).mapToInt(x -> x).toArray();
            if (predictionList[0] == actualCategory) {
              labeledObject.freeRef();
              tensor.freeRef();
              return null; // We will only examine mispredicted rows
            }
            @Nonnull final LinkedHashMap<CharSequence, Object> row = new LinkedHashMap<>();
            row.put("Image", log.png(labeledObject.data.toGrayImage(), labeledObject.label));
            labeledObject.freeRef();
            row.put("Prediction",
                RefUtil.get(RefArrays.stream(predictionList).limit(3)
                    .mapToObj(i -> RefString.format("%d (%.1f%%)", i, 100.0 * tensor.get(i)))
                    .reduce((a, b) -> a + ", " + b)));
            tensor.freeRef();
            return row;
          }, recognitionNetwork.addRef()))
          .filter(x -> null != x)
          .limit(10)
          .forEach(properties -> table.putRow(properties));
      return table;
    }, recognitionNetwork.addRef()));
    recognitionNetwork.freeRef();
  }

  public int parse(@Nonnull final String label) {
    return Integer.parseInt(label.replaceAll("[^\\d]", ""));
  }

  public int[] predict(@Nonnull final Layer network, @Nonnull final LabeledObject<Tensor> labeledObject) {
    Result temp_06_0019 = network.eval(labeledObject.data.addRef());
    assert temp_06_0019 != null;
    TensorList temp_06_0020 = temp_06_0019.getData();
    Tensor tensor = temp_06_0020.get(0);
    temp_06_0020.freeRef();
    temp_06_0019.freeRef();
    network.freeRef();
    labeledObject.freeRef();
    int[] temp_06_0007 = RefIntStream.range(0, 10).mapToObj(x -> x)
        .sorted(RefComparator
            .comparingDouble(RefUtil.wrapInterface(i -> -tensor.get(i),
                tensor.addRef())))
        .mapToInt(x -> x).toArray();
    tensor.freeRef();
    return temp_06_0007;
  }

  protected abstract Layer buildModel(@Nonnull NotebookOutput log);
}
