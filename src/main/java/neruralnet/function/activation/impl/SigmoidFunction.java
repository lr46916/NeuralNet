package neruralnet.function.activation.impl;

import java.io.Serializable;

import neruralnet.function.activation.ActivationFun;

public class SigmoidFunction implements ActivationFun, Serializable {

	private static final long serialVersionUID = -1549340712794373270L;

	@Override
	public double apply(double x) {
		return 1. / (1 + Math.pow(Math.E, -x));
	}

}
