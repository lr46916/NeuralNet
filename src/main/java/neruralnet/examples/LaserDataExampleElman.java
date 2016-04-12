//package neruralnet.examples;
//
//import java.io.BufferedInputStream;
//import java.io.BufferedReader;
//import java.io.FileInputStream;
//import java.io.IOException;
//import java.io.InputStreamReader;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.List;
//
//import hr.fer.zemris.optim.evol.Crossover;
//import hr.fer.zemris.optim.evol.Evaluator;
//import hr.fer.zemris.optim.evol.Mutation;
//import hr.fer.zemris.optim.evol.algorithms.ga.GeneticAlgorithm;
//import hr.fer.zemris.optim.evol.algorithms.ga.tournament.EliminationGA;
//import hr.fer.zemris.optim.evol.chromosome.FloatingPointChromosome;
//import hr.fer.zemris.optim.evol.crossovers.impl.floatingpoint.FPSimpleCrossover;
//import hr.fer.zemris.optim.evol.mutations.impl.folatingpoint.FPGaussianMutation;
//import hr.fer.zemris.optim.evol.mutations.impl.permutation.MultipleChoiceMutation;
//import hr.fer.zemris.optim.evol.populationgenerator.PopulationGenerator;
//import hr.fer.zemris.optim.evol.populationgenerator.impl.FloatingPointChromosomePG;
//import hr.fer.zemris.optim.evol.selection.impl.KTournamentSelection;
//import hr.fer.zemris.optim.evol.selection.impl.SelectionTournament;
//import hr.fer.zemris.optim.rng.RNG;
//import neruralnet.StatefullLayeredNetwork;
//import neruralnet.function.activation.ActivationFun;
//import neruralnet.layer.Layer;
//import neruralnet.layer.impl.ElmanLayer;
//import neruralnet.layer.impl.FullyConnected;
//
//public class LaserDataExampleElman {
//
//	private static double[] loadLasetData(String fileName) throws IOException {
//		BufferedReader br = new BufferedReader(
//				new InputStreamReader(new BufferedInputStream(new FileInputStream(fileName))));
//		double[] data = new double[1000];
//		String line = null;
//		int ind = 0;
//		while ((line = br.readLine()) != null) {
//			data[ind++] = Double.parseDouble(line);
//		}
//		br.close();
//		return data;
//	}
//
//	private static double[] normalizeData(double[] data) {
//		double[] normalized = Arrays.copyOf(data, data.length);
//		double max = Double.MIN_VALUE;
//		double min = Double.MAX_VALUE;
//		for (int i = 0; i < data.length; ++i) {
//			max = Double.max(max, data[i]);
//			min = Double.min(min, data[i]);
//		}
//		for (int i = 0; i < data.length; ++i) {
//			normalized[i] = ((normalized[i] - min) / (max - min)) * 2 - 1;
//		}
//		return normalized;
//	}
//
//	private static StatefullLayeredNetwork createElmanNetwork(String configuration) {
//		String[] inputsOutputs = configuration.split("[x]");
//		List<Layer> layers = new ArrayList<>();
//		for (int i = 0; i < inputsOutputs.length - 1; ++i) {
//			int in = Integer.parseInt(inputsOutputs[i]);
//			int out = Integer.parseInt(inputsOutputs[i + 1]);
//			if (i == 0) {
//				layers.add(new ElmanLayer(in, out, ActivationFun.tangh));
//			} else
//				layers.add(new FullyConnected(in, out, ActivationFun.tangh));
//		}
//		return new StatefullLayeredNetwork(layers);
//	}
//
//	public static void main(String[] args) throws IOException {
//
//		double[] laserData = loadLasetData("lib/laser-data.txt");
//
//		double[] normalized = normalizeData(laserData);
//
//		Crossover<FloatingPointChromosome> crossover = new FPSimpleCrossover();
//
//		Mutation<FloatingPointChromosome> mutation = new FPGaussianMutation(0, 0.1, 0.01);
//		Mutation<FloatingPointChromosome> flipMutation = (target) -> {
//			target.data[RNG.getRNG().nextInt(target.data.length)] *= -1;
//		};
//		@SuppressWarnings("unchecked")
//		Mutation<FloatingPointChromosome>[] mutations = (Mutation<FloatingPointChromosome>[]) (new Mutation<?>[] {
//				mutation, flipMutation });
//		double[] poss = new double[] { 0.8, 0.2 };
//		MultipleChoiceMutation<FloatingPointChromosome> multipleMutation = new MultipleChoiceMutation<>(mutations,
//				poss);
//
//		StatefullLayeredNetwork ffn = createElmanNetwork("1x10x4x1");
//
//		PopulationGenerator<FloatingPointChromosome> pg = new FloatingPointChromosomePG(
//				ffn.getNumberOfWeights() + ffn.getContextSize());
//
//		Evaluator<FloatingPointChromosome> evaluator = new ElmanEvaluator(Arrays.copyOf(normalized, 600), ffn);
//
//		SelectionTournament selection = new KTournamentSelection(3);
//
//		GeneticAlgorithm<FloatingPointChromosome> ga = new EliminationGA<FloatingPointChromosome>(100, pg, crossover,
//				multipleMutation, evaluator, 60000, selection);
//
//		FloatingPointChromosome best = ga.run();
//
//		ffn.setWeigths(best.data, 0);
//		ffn.setContext(best.data, ffn.getNumberOfWeights());
//
//		double[] results = Arrays.copyOf(normalized, normalized.length);
//
//		double[] inputs = new double[ffn.getNumberOfInputs()];
//		double[] outputs = new double[ffn.getNumberOfOutputs()];
//		double error = 0;
//		for (int i = 1; i < 1000; ++i) {
//			inputs[0] = normalized[i - 1];
//			ffn.apply(inputs, outputs);
//			double err = normalized[i] - outputs[0];
//			error += err * err;
//			results[i] = outputs[0];
//		}
//		System.out.println(error);
//		renormalizeAndPrint(laserData, results);
//	}
//
//	private static void renormalizeAndPrint(double[] truth, double[] predictions) {
//		double max = Double.MIN_VALUE;
//		double min = Double.MAX_VALUE;
//		for (int i = 0; i < truth.length; ++i) {
//			max = Double.max(max, truth[i]);
//			min = Double.min(min, truth[i]);
//		}
//
//		for (int i = 0; i < truth.length; ++i) {
//			double renorm = (predictions[i] + 1) * 0.5 * (max - min) + min;
//			System.out.format("%d: Truth: %.3f, prediction: %.3f%n", i + 1, truth[i], renorm);
//		}
//
//	}
//
//	private static class ElmanEvaluator implements Evaluator<FloatingPointChromosome> {
//		private double[] data;
//		private StatefullLayeredNetwork ffn;
//		private double[] inputs, outputs;
//
//		public ElmanEvaluator(double[] data, StatefullLayeredNetwork ffn) {
//			super();
//			this.data = data;
//			this.ffn = ffn;
//			this.inputs = new double[ffn.getNumberOfInputs()];
//			this.outputs = new double[ffn.getNumberOfOutputs()];
//		}
//
//		@Override
//		public void evaluate(FloatingPointChromosome target) {
//			ffn.setWeigths(target.data, 0);
//			ffn.setContext(target.data, ffn.getNumberOfWeights());
//			double error = 0;
//			for (int i = 1; i < data.length; ++i) {
//				inputs[0] = data[i - 1];
//				ffn.apply(inputs, outputs);
//				double err = data[i] - outputs[0];
//				error += err * err;
//			}
//			// target.fitness = -(error / (data.length - 1));
//			target.fitness = -error;
//		}
//	}
//
//}
