package neruralnet.function.activation.impl;

import neruralnet.function.activation.ActivationFunBP;

public class ReLU implements ActivationFunBP {

	private static final double LEAKY_FACTOR = 0.01;
	
	@Override
	public double apply(double value) {
		return Double.max(LEAKY_FACTOR, value);
	}

	@Override
	public double gradient(double x) {
		return x > 0 ? 1 : LEAKY_FACTOR; //leaky implementation to avoid dying reLU neurons
	}

}
