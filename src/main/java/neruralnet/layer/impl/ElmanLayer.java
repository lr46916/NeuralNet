package neruralnet.layer.impl;

import neruralnet.function.distribution.Distribution;

public class ElmanLayer extends FullyConnected {

	private double[] extendedInput;

	public ElmanLayer(int n, int m, Distribution dist) {
		super(n + m, m, dist);
		extendedInput = new double[n + m];
	}

	@Override
	public void apply(double[] inputs, double[] outputs) {
		for (int i = 0; i < inputs.length; i++) {
			extendedInput[i] = inputs[i];
		}
		super.apply(extendedInput, outputs);
		for (int i = 0; i < outputs.length; i++) {
			extendedInput[i + inputs.length] = outputs[i];
		}
	}

}
