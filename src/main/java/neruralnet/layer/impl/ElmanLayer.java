package neruralnet.layer.impl;

import java.io.Serializable;
import java.util.Arrays;

import neruralnet.function.activation.ActivationFun;
import neruralnet.function.distribution.Distribution;
import neruralnet.function.distribution.impl.NormalDist;
import neruralnet.layer.StatefulLayer;
import neruralnet.layer.activation.ActivationFunLayer;

public class ElmanLayer extends StatefulLayer implements Serializable{

	private static final long serialVersionUID = 1706291596813401214L;
	
	private double[] extendedInput;
	private int n;
	private int m;
	private FullyConnected fullyConnected;
	private ActivationFunLayer activationFun;

	public ElmanLayer(int n, int m, Distribution dist, ActivationFun activation) {
		this.n = n;
		this.m = m;
		extendedInput = new double[n + m];
		fullyConnected = new FullyConnected(n + m, m, dist);
		activationFun = new ActivationFunLayer(activation, m);
	}

	public ElmanLayer(int n, int m, ActivationFun activation) {
		this.n = n;
		this.m = m;
		extendedInput = new double[n + m];
		fullyConnected = new FullyConnected(n + m, m);
		activationFun = new ActivationFunLayer(activation, m);
	}

	@Override
	public void apply(double[] inputs, double[] outputs) {
		for (int i = 0; i < inputs.length; i++) {
			extendedInput[i + m] = inputs[i];
		}
		fullyConnected.apply(extendedInput, outputs);
		activationFun.apply(outputs, outputs);
		for (int i = 0; i < outputs.length; i++) {
			if (Double.isNaN(outputs[i])) {
				System.exit(-1);
			}
			extendedInput[i] = outputs[i];
		}
	}

	@Override
	public int getNumberOfInputs() {
		return n;
	}

	@Override
	public void setWeigths(double[] weights, int offset) {
		fullyConnected.setWeigths(weights, offset);
	}

	@Override
	public double[] getWeights() {
		return fullyConnected.getWeights();
	}

	@Override
	public int getNumberOfOutputs() {
		return m;
	}

	@Override
	public int getNumberOfWeights() {
		return fullyConnected.getNumberOfWeights();
	}

	@Override
	public double[] getContext() {
		double[] ret = new double[m];
		for (int i = 0; i < m; i++) {
			ret[i] = extendedInput[i];
		}
		return ret;
	}

	@Override
	public void setContext(double[] context, int offset) {
		for (int i = 0; i < m; i++) {
			extendedInput[i] = context[i + offset];
		}
	}
	
	@Override
	public int getContextSize() {
		return m;
	}
	
	public static void main(String[] args) {
		ElmanLayer el = new ElmanLayer(3, 1, new NormalDist(0, 1), ActivationFun.sigmoid);

		double[] inputs = new double[] { 0.1, 0.2, 0.3 };
		double[] outputs = new double[1];

		el.apply(inputs, outputs);

		System.out.println(Arrays.toString(outputs));
		System.out.println(Arrays.toString(el.extendedInput));

		el.apply(inputs, outputs);

		System.out.println(Arrays.toString(outputs));

		System.out.println(Arrays.toString(el.extendedInput));
	}

}
