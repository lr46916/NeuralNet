package neruralnet.function.activation;

public interface ActivationFun {
	
	public static ActivationFun sigmoid = (x) -> 1. / (1 + Math.pow(Math.E, -x));
	
	public double apply(double value);
}
