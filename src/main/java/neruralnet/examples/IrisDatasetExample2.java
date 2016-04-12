package neruralnet.examples;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import neruralnet.NeuralNetUtil;
import neruralnet.StatelessLayeredNeuralNet;
import neruralnet.function.activation.ActivationFun;
import neruralnet.function.activation.ActivationFunBP;
import neruralnet.layer.BPLayer;
import neruralnet.learning.BackPropagation;
import neruralnet.sample.DataSample;

public class IrisDatasetExample2 {

	public static void main(String[] args) throws IOException {

		StatelessLayeredNeuralNet ffn = NeuralNetUtil.createNetwork("4x5x3x3", (n) -> {
			Random rand = new Random();
			double[] res = new double[n];
			for (int i = 0; i < n; ++i) {
				res[i] = rand.nextDouble() - 1;
			}
			return res;
		} , ActivationFun.sigmoid, ActivationFun.sigmoid, ActivationFun.sigmoid);

		IrisDataSetEvaluator evaluator = new IrisDataSetEvaluator("lib/iris.txt");

		BPLayer[] layers = new BPLayer[ffn.getLayers().length];
		ActivationFunBP[] activations = new ActivationFunBP[ffn.getActivations().length];

		for (int i = 0; i < layers.length; ++i) {
			layers[i] = (BPLayer) ffn.getLayers()[i];
			activations[i] = (ActivationFunBP) ffn.getActivations()[i];
		}

		BackPropagation backprop = new BackPropagation(layers, activations);
		
		backprop.trainBackpropagation(evaluator.train, 50000, 10, 1000, 0.1);

		int wrong = 0;
		double[] out = new double[3];
		for (DataSample t : evaluator.test) {

			ffn.apply(t.inputs, out);

			for (int i = 0; i < 3; i++) {
				out[i] = out[i] >= 0.5 ? 1 : 0;
			}

			if (!Arrays.equals(out, t.outputs)) {
				wrong++;
			}
		}
		System.out.println("Classification accuracy on test: "
				+ ((double) (evaluator.test.size() - wrong) / evaluator.test.size()));
	}

	private static class IrisDataSetEvaluator {

		public List<DataSample> train;
		public List<DataSample> test;

		public IrisDataSetEvaluator(String filename) throws IOException {
			BufferedReader br = new BufferedReader(
					new InputStreamReader(new BufferedInputStream(new FileInputStream(filename))));

			String line = null;

			List<DataSample> data = new ArrayList<>();

			while ((line = br.readLine()) != null) {

				String[] parts = line.split("[:]");

				String input = parts[0].substring(1, parts[0].length() - 1);
				String output = parts[1].substring(1, parts[1].length() - 1);

				String[] inVal = input.split("[,]");
				String[] outVal = output.split("[,]");

				double[] in = new double[inVal.length];
				double[] out = new double[outVal.length];

				for (int i = 0; i < 4; i++) {
					in[i] = Double.parseDouble(inVal[i]);
				}
				for (int i = 0; i < 3; i++) {
					out[i] = Double.parseDouble(outVal[i]);
				}

				data.add(new DataSample(in, out));
			}

			Collections.shuffle(data);
			int ts = (int) (data.size() * 0.8);
			train = data.subList(0, ts);
			test = data.subList(ts + 1, data.size());

			// System.out.println(test);
			//
			// System.exit(-1);

			br.close();
		}

	}

}
