package neruralnet.function.activation;

public interface ActivationFun {

	public static ActivationFun sigmoid = (x) -> 1. / (1 + Math.pow(Math.E, -x));
	public static ActivationFun tangh = (x) -> {
		double tmp = Math.pow(Math.E, -2 * x);
		return (1. - tmp) / (1. + tmp);
	};

	public double apply(double value);
}
