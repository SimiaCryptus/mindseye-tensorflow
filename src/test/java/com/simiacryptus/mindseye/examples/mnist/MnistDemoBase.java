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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.tensorflow.TensorboardEventWriter;
import com.simiacryptus.util.CodeUtil;
import com.simiacryptus.util.test.LabeledObject;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.framework.GraphDef;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.function.IntFunction;

public abstract @RefAware
class MnistDemoBase {
  private static final Logger log = LoggerFactory.getLogger(MnistDemoBase.class);
  protected int timeout = 60;

  protected abstract byte[] getGraphDef();

  @Test
  public void test() throws Exception {
    File tensorboardLocation = new File("tensorboard/tflayer_TFLayer").getAbsoluteFile();
    File[] listFiles = tensorboardLocation.getParentFile().listFiles();
    if (null != listFiles)
      RefArrays.stream(listFiles).forEach(file -> {
        com.simiacryptus.ref.wrappers.RefSystem.out.println("Delete: " + file);
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
    if (null != graphDef) {
      TFUtil.launchTensorboard(tensorboardLocation.getParentFile(), x -> {
        JOptionPane.showConfirmDialog(null, "Press OK to exit");
      });
    }
  }

  public void run(@Nonnull NotebookOutput log) {

    final Tensor[][] trainingData = MNIST.trainingDataStream().map(labeledObject1 -> {
      @Nonnull final Tensor categoryTensor = new Tensor(10);
      final int category = parse(labeledObject1.label);
      RefUtil.freeRef(categoryTensor.set(category, 1));
      Tensor[] temp_06_0001 = new Tensor[]{labeledObject1.data,
          categoryTensor == null ? null : categoryTensor};
      return temp_06_0001;
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
      public void onStepComplete(final Step currentPoint) {
        if (null != TFLayerBase.eventWriter) {
          TFLayerBase.eventWriter.setStep(currentPoint.iteration);
        }
        history.add(currentPoint == null ? null : currentPoint.addRef());
        super.onStepComplete(currentPoint);
        if (null != currentPoint)
          currentPoint.freeRef();
      }
    };
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<Double>) () -> {
              EntropyLossLayer loss = new EntropyLossLayer();
              @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(
                  recognitionNetwork == null ? null : recognitionNetwork.addRef(), loss == null ? null : loss.addRef());
              if (null != loss)
                loss.freeRef();
              @Nonnull final Trainable trainable = new SampledArrayTrainable(
                  Tensor.addRefs(trainingData),
                  supervisedNetwork == null ? null : supervisedNetwork, 1000, 1000);
              IterativeTrainer temp_06_0008 = new IterativeTrainer(
                  trainable == null ? null : trainable);
              IterativeTrainer temp_06_0009 = temp_06_0008.setMonitor(monitor);
              IterativeTrainer temp_06_0010 = temp_06_0009.setOrientation(new LBFGS());
              IterativeTrainer temp_06_0011 = temp_06_0010
                  //          .setLineSearchFactory(n -> new ArmijoWolfeSearch().setAlpha(1e0))
                  .setLineSearchFactory(n -> new QuadraticSearch());
              IterativeTrainer temp_06_0012 = temp_06_0011.setTimeout(timeout,
                  TimeUnit.SECONDS);
              IterativeTrainer temp_06_0013 = temp_06_0012.setMaxIterations(200);
              IterativeTrainer temp_06_0014 = temp_06_0013.setIterationsPerSample(20);
              double temp_06_0002 = temp_06_0014.run();
              if (null != temp_06_0014)
                temp_06_0014.freeRef();
              if (null != temp_06_0013)
                temp_06_0013.freeRef();
              if (null != temp_06_0012)
                temp_06_0012.freeRef();
              if (null != temp_06_0011)
                temp_06_0011.freeRef();
              if (null != temp_06_0010)
                temp_06_0010.freeRef();
              if (null != temp_06_0009)
                temp_06_0009.freeRef();
              if (null != temp_06_0008)
                temp_06_0008.freeRef();
              //          .setLineSearchFactory(n -> new ArmijoWolfeSearch().setAlpha(1e0))
              return temp_06_0002;
            }, Tensor.addRefs(trainingData),
            recognitionNetwork == null ? null : recognitionNetwork.addRef()));
    if (null != trainingData)
      ReferenceCounting.freeRefs(trainingData);
    if (!history.isEmpty()) {
      log.eval(RefUtil
          .wrapInterface((UncheckedSupplier<PlotCanvas>) () -> {
            @Nonnull final PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> {
              double[] temp_06_0003 = new double[]{step.iteration, Math.log10(step.point.getMean())};
              if (null != step)
                step.freeRef();
              return temp_06_0003;
            }).toArray(i -> new double[i][]));
            plot.setTitle("Convergence Plot");
            plot.setAxisLabels("Iteration", "log10(Fitness)");
            plot.setSize(600, 400);
            return plot;
          }, history == null ? null : history.addRef()));
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
    log.p("Saved model as " + log.file(recognitionNetwork.getJson().toString(), modelName, modelName));

    if (null != TFLayerBase.eventWriter) {
      try {
        TFLayerBase.eventWriter.close();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      TFLayerBase.eventWriter = null;
    }

    log.h1("Validation");
    log.p("If we apply our model against the entire validation dataset, we get this accuracy:");
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<Double>) () -> {
          RefList<LabeledObject<Tensor>> validation = MNIST.validationDataStream().collect(RefCollectors.toList());
          Tensor[][] tensors = new Tensor[][]{validation.stream().map(x -> x.data).toArray(i -> new Tensor[i])};
          Result temp_06_0015 = recognitionNetwork
              .eval(Tensor.addRefs(tensors));
          TensorList predictionData = temp_06_0015.getData();
          if (null != temp_06_0015)
            temp_06_0015.freeRef();
          if (null != tensors)
            ReferenceCounting.freeRefs(tensors);
          RefList<int[]> predicitonList = RefIntStream.range(0, predictionData.length())
              .mapToObj(RefUtil
                  .wrapInterface((IntFunction<? extends int[]>) rowIndex -> {
                    Tensor predictionTensor = predictionData.get(rowIndex);
                    int[] temp_06_0005 = RefIntStream.range(0, 10).mapToObj(x -> x)
                        .sorted(RefComparator.comparing(RefUtil.wrapInterface(
                            (Function<? super Integer, ? extends Double>) ii -> {
                              return -predictionTensor.getData()[ii];
                            }, predictionTensor == null ? null : predictionTensor.addRef())))
                        .mapToInt(x -> x).toArray();
                    if (null != predictionTensor)
                      predictionTensor.freeRef();
                    return temp_06_0005;
                  }, predictionData == null ? null : predictionData.addRef()))
              .collect(RefCollectors.toList());
          double temp_06_0004 = RefIntStream.range(0, predictionData.length()).mapToDouble(
              RefUtil.wrapInterface(rowIndex -> {
                    return predicitonList.get(rowIndex)[0] == parse(validation.get(rowIndex).label) ? 1 : 0;
                  }, validation == null ? null : validation.addRef(),
                  predicitonList == null ? null : predicitonList.addRef()))
              .average().getAsDouble() * 100;
          if (null != predicitonList)
            predicitonList.freeRef();
          if (null != predictionData)
            predictionData.freeRef();
          if (null != validation)
            validation.freeRef();
          return temp_06_0004;
        }, recognitionNetwork == null ? null : recognitionNetwork.addRef()));

    log.p("Let's examine some incorrectly predicted results in more detail:");
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<TableOutput>) () -> {
          @Nonnull final TableOutput table = new TableOutput();
          MNIST.validationDataStream().map(RefUtil.wrapInterface(
              (Function<? super LabeledObject<Tensor>, ? extends RefLinkedHashMap<CharSequence, Object>>) labeledObject -> {
                final int actualCategory = parse(labeledObject.label);
                Result temp_06_0016 = recognitionNetwork
                    .eval(labeledObject.data.addRef());
                TensorList temp_06_0017 = temp_06_0016.getData();
                Tensor temp_06_0018 = temp_06_0017.get(0);
                @Nullable final double[] predictionSignal = temp_06_0018.getData();
                if (null != temp_06_0018)
                  temp_06_0018.freeRef();
                if (null != temp_06_0017)
                  temp_06_0017.freeRef();
                if (null != temp_06_0016)
                  temp_06_0016.freeRef();
                final int[] predictionList = RefIntStream.range(0, 10).mapToObj(x -> x)
                    .sorted(RefComparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
                if (predictionList[0] == actualCategory)
                  return null; // We will only examine mispredicted rows
                @Nonnull final RefLinkedHashMap<CharSequence, Object> row = new RefLinkedHashMap<>();
                row.put("Image", log.png(labeledObject.data.toGrayImage(), labeledObject.label));
                row.put("Prediction",
                    RefArrays.stream(predictionList).limit(3)
                        .mapToObj(i -> RefString.format("%d (%.1f%%)", i, 100.0 * predictionSignal[i]))
                        .reduce((a, b) -> a + ", " + b).get());
                return row;
              }, recognitionNetwork == null ? null : recognitionNetwork.addRef())).filter(x -> {
            boolean temp_06_0006 = null != x;
            if (null != x)
              x.freeRef();
            return temp_06_0006;
          }).limit(10).forEach(table::putRow);
          return table;
        }, recognitionNetwork == null ? null : recognitionNetwork.addRef()));
    if (null != recognitionNetwork)
      recognitionNetwork.freeRef();

  }

  public int parse(@Nonnull final String label) {
    return Integer.parseInt(label.replaceAll("[^\\d]", ""));
  }

  public int[] predict(@Nonnull final Layer network, @Nonnull final LabeledObject<Tensor> labeledObject) {
    Result temp_06_0019 = network.eval(labeledObject.data.addRef());
    TensorList temp_06_0020 = temp_06_0019.getData();
    Tensor tensor = temp_06_0020.get(0);
    if (null != temp_06_0020)
      temp_06_0020.freeRef();
    if (null != temp_06_0019)
      temp_06_0019.freeRef();
    network.freeRef();
    int[] temp_06_0007 = RefIntStream.range(0, 10).mapToObj(x -> x)
        .sorted(RefComparator.comparing(RefUtil.wrapInterface(
            (Function<? super Integer, ? extends Double>) i -> -tensor
                .getData()[i],
            tensor == null ? null : tensor.addRef())))
        .mapToInt(x -> x).toArray();
    if (null != tensor)
      tensor.freeRef();
    return temp_06_0007;
  }

  protected abstract Layer buildModel(@Nonnull NotebookOutput log);
}
