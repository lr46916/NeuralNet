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
import java.util.stream.Collector;
import java.util.stream.Collectors;

import hr.fer.zemris.optim.algorithms.evol.ga.GeneticAlgorithm;
import hr.fer.zemris.optim.algorithms.evol.ga.tournament.EliminationGA;
import hr.fer.zemris.optim.evol.Crossover;
import hr.fer.zemris.optim.evol.Evaluator;
import hr.fer.zemris.optim.evol.Mutation;
import hr.fer.zemris.optim.evol.chromosome.FloatingPointChromosome;
import hr.fer.zemris.optim.evol.crossovers.impl.floatingpoint.FPSimpleCrossover;
import hr.fer.zemris.optim.evol.mutations.impl.folatingpoint.FPGaussianMutation;
import hr.fer.zemris.optim.evol.populationgenerator.PopulationGenerator;
import hr.fer.zemris.optim.evol.populationgenerator.impl.FloatingPointChromosomePG;
import hr.fer.zemris.optim.evol.selection.impl.KTournamentSelection;
import hr.fer.zemris.optim.evol.selection.impl.SelectionTournament;
import neruralnet.FFNeuralNet;
import neruralnet.function.activation.ActivationFun;
import neruralnet.function.distribution.Distribution;
import neruralnet.function.distribution.impl.NormalDist;
import neruralnet.layer.Layer;
import neruralnet.layer.activation.ActivationFunLayer;
import neruralnet.layer.impl.FullyConnected;

public class IrisDatasetExample {

	public static void main(String[] args) throws IOException {

		SelectionTournament selection = new KTournamentSelection(3);

		Crossover<FloatingPointChromosome> crossover = new FPSimpleCrossover();

		Mutation<FloatingPointChromosome> mutation = new FPGaussianMutation(0, 1, 0.01);

		Distribution dist = new NormalDist(0, 1);
		
		FFNeuralNet ffn = new FFNeuralNet(new FullyConnected(4, 5, dist), new ActivationFunLayer(ActivationFun.sigmoid, 5), new FullyConnected(5, 3, dist),new ActivationFunLayer(ActivationFun.sigmoid, 3)
				, new FullyConnected(3, 3, dist), new ActivationFunLayer(ActivationFun.sigmoid, 3));
		
		PopulationGenerator<FloatingPointChromosome> pg = new FloatingPointChromosomePG(ffn.getNumberOfWeights());
		
		IrisDataSetEvaluator evaluator = new IrisDataSetEvaluator("lib/iris.txt", ffn);

		GeneticAlgorithm<FloatingPointChromosome> ga = new EliminationGA<>(50, pg, crossover, mutation, evaluator,
				10000 * 5, selection);

		FloatingPointChromosome res = ga.run();

		System.out.println("Result: " + res.fitness);
		
		int wrong = 0;
		
		ffn.setWeigths(res.data, 0);
		double[] out = new double[3];
		for(neruralnet.examples.IrisDatasetExample.IrisDataSetEvaluator.Tuple t : evaluator.test) {
			
			ffn.apply(t.inputs, out);
			
			for(int i = 0; i < 3; i++) {
				out[i] = out[i] > 0.5 ? 1 : 0;
			}
			
			if(!Arrays.equals(out, t.out)) {
				wrong++;
			}
		}
		
		System.out.println(wrong);
		System.out.println(evaluator.test.size());

	}

	private static class IrisDataSetEvaluator implements Evaluator<FloatingPointChromosome> {

		private List<Tuple> train;
		public List<Tuple> test;
		private FFNeuralNet net;

		public IrisDataSetEvaluator(String filename, FFNeuralNet net) throws IOException {
			this.net = net;
			BufferedReader br = new BufferedReader(
					new InputStreamReader(new BufferedInputStream(new FileInputStream(filename))));

			String line = null;

			List<Tuple> data = new ArrayList<>();

			while ((line = br.readLine()) != null) {

				String[] parts = line.split("[:]");

				String input = parts[0].substring(1, parts[0].length() - 1);
				String output = parts[1].substring(1, parts[1].length() - 1);

				String[] inVal = input.split("[,]");
				String[] outVal = output.split("[,]");

				double[] in = new double[inVal.length];
				double[] out = new double[outVal.length];
				
				for(int i = 0; i < 4; i++) {
					in[i] = Double.parseDouble(inVal[i]);
				}
				for(int i = 0; i < 3; i++) {
					out[i] = Double.parseDouble(outVal[i]);
				}
				
				
				data.add(new Tuple(in, out));
			}

			Collections.shuffle(data);
			int ts = (int) (data.size() * 0.8);
			train = data.subList(0, ts);
			test = data.subList(ts + 1, data.size());
			
//			System.out.println(test);
//			
//			System.exit(-1);

			br.close();
		}

		@Override
		public void evaluate(FloatingPointChromosome arg0) {
			double[] ws = arg0.data;
			double[] netOut = new double[3];
			net.setWeigths(ws, 0);
			double error = 0;
			for (Tuple t : train) {
				net.apply(t.inputs, netOut);
				for (int i = 0; i < 3; i++) {
					error += (netOut[i] - t.out[i]) * (netOut[i] - t.out[i]);
				}
			}
			arg0.fitness = -error;
		}

		public static class Tuple {
			public double[] inputs;
			public double[] out;

			public Tuple(double[] inputs, double[] out) {
				super();
				this.inputs = inputs;
				this.out = out;
			}
			
			public String toString(){
				return "inputs: " + Arrays.toString(inputs) + ", out: " + Arrays.toString(out);
			}
		}

	}

}
