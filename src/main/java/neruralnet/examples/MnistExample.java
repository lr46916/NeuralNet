package neruralnet.examples;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import neruralnet.NeuralNetUtil;
import neruralnet.StatelessLayeredNeuralNet;
import neruralnet.examples.dataloader.DigitImage;
import neruralnet.examples.dataloader.DigitImageLoadingService;
import neruralnet.function.activation.ActivationFun;
import neruralnet.function.activation.ActivationFunBP;
import neruralnet.function.distribution.impl.NormalDist;
import neruralnet.layer.BPLayer;
import neruralnet.learning.BackPropagation;
import neruralnet.sample.DataSample;

public class MnistExample {
	private static List<DataSample> loadData(String labelsPath, String imgPath) throws IOException {
		List<DigitImage> images = DigitImageLoadingService.loadDigitImages(labelsPath, imgPath);
		List<DataSample> trainData = new ArrayList<>(images.size());

		for (int i = 0; i < images.size(); ++i) {
			double[] inputs = new double[784];
			for (int j = 0; j < inputs.length; ++j) {
				byte tmp = images.get(i).imageData[j];
				inputs[j] = Byte.toUnsignedInt(tmp) / 255.;
			}
			double[] outputs = new double[10];
			outputs[images.get(i).label] = 1;
			trainData.add(new DataSample(inputs, outputs));
		}
		return trainData;
	}

	public static void main(String[] args) throws IOException {
		List<DataSample> trainData = loadData("lib/train-labels.idx1-ubyte", "lib/train-images.idx3-ubyte");

		StatelessLayeredNeuralNet ffn = NeuralNetUtil.createNetwork("784x30x10", (n) -> {
			Random rand = new Random();
			double[] res = new double[n];
			for (int i = 0; i < n; ++i) {
				res[i] = 2 * rand.nextDouble() - 1;
			}
			return res;
		} , ActivationFun.sigmoid, ActivationFun.sigmoid);

		BPLayer[] layers = new BPLayer[ffn.getLayers().length];
		ActivationFunBP[] activations = new ActivationFunBP[ffn.getActivations().length];

		for (int i = 0; i < layers.length; ++i) {
			layers[i] = (BPLayer) ffn.getLayers()[i];
			activations[i] = (ActivationFunBP) ffn.getActivations()[i];
		}

		BackPropagation backprop = new BackPropagation(layers, activations);

		backprop.trainBackpropagation(trainData, 30, 100, 1, 10);

		System.out.println("Train set performance: " + performance(ffn, trainData));
		System.out.println("Test set performance: "
				+ performance(ffn, loadData("lib/t10k-labels.idx1-ubyte", "lib/t10k-images.idx3-ubyte")));

	}

	private static double performance(StatelessLayeredNeuralNet ffn, List<DataSample> data) {

		int hit = 0;
		double[] outputs = new double[ffn.getNumberOfOutputs()];
		for (DataSample sample : data) {
			ffn.apply(sample.inputs, outputs);
			int maxInd = 0;
			double maxVal = outputs[0];
			for (int i = 1; i < outputs.length; ++i) {
				if (maxVal < outputs[i]) {
					maxInd = i;
					maxVal = outputs[i];
				}
			}
			if (sample.outputs[maxInd] == 1) {
				hit++;
			}
		}
		return (double) hit / data.size();
	}
}
