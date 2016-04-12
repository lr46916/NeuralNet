package neruralnet.function.activation.impl;

import java.io.Serializable;

import neruralnet.function.activation.ActivationFunBP;

public class SigmoidFunction implements ActivationFunBP, Serializable {

	private static final long serialVersionUID = -1549340712794373270L;

	@Override
	public double apply(double x) {
		return 1 / (1 + Math.expm1(-x) + 1);
	}
	
	@Override
	public double gradient(double x) {
		double y = apply(x);
		return (1 - y) * y;
	}

}
