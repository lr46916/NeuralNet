package neruralnet;

import neruralnet.function.activation.ActivationFun;
import neruralnet.layer.PrototypeLayer;

public class StatefullLayeredNetworkProto extends StatefullLayeredNetwork implements PrototypeLayer {

	private static final long serialVersionUID = -6178369450429870327L;
	private PrototypeLayer[] layers;
	private ActivationFun[] activation;
	
	public StatefullLayeredNetworkProto(PrototypeLayer[] layers, ActivationFun[] activation) {
		super(layers, activation);
		this.layers = layers;
		this.activation = activation;
	}
	
	@Override
	public PrototypeLayer duplicate() {
		PrototypeLayer[] clones = new PrototypeLayer[layers.length];
		for(int i = 0; i < clones.length; i++) {
			clones[i] = layers[i].duplicate();
		}
		
		return new StatefullLayeredNetworkProto(clones, activation);
	}

}
