package neruralnet.examples.dataloader;

public class DigitImage {
	public int label;
	public byte[] imageData;

	public DigitImage(int label, byte[] imageData) {
		super();
		this.label = label;
		this.imageData = imageData;
	}


}
