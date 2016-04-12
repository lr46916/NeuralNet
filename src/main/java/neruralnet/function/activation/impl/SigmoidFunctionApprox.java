package neruralnet.function.activation.impl;

import java.io.Serializable;

import neruralnet.function.activation.ActivationFunBP;

public class SigmoidFunctionApprox implements ActivationFunBP, Serializable {

	private static final long serialVersionUID = 7435848420849546578L;
	
	private static double factor = 1 / Math.log(2);

	private static double exp(double val) {
		if (val > 40) {
			return Double.MAX_VALUE;
		}
		double exponent = val * factor;
		int integerPart = (int) exponent;
		return (1L << integerPart) * Math.pow(2, exponent - integerPart);
	}

	@Override
	public double apply(double value) {
		if (value > 0) {
			return 1. / (1 + exp(value));
		} else {
			return 1. / (1 + 1 / exp(-value));
		}
	}
	
	@Override
	public double gradient(double x) {
		double y = apply(x);
		return (1 - y) * y;
	}

}
