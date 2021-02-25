
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.*;
import org.tensorflow.types.TFloat32;

import java.io.File;

import org.tensorflow.SavedModelBundle;

public class Demo {
	public static void main(String[] args) {
		String filePath = new File("").getAbsolutePath();
		filePath = filePath + "/src/main/resources/f_model/";
		System.out.println(filePath);
		SavedModelBundle model = SavedModelBundle.load(filePath);
		NdArray<Float> input_matrix = NdArrays.ofFloats(Shape.of(1, 10));
		Tensor input_tensor = TFloat32.tensorOf(input_matrix);
		Tensor out = model.function("serving_default").call(input_tensor);
		System.out.println(out.shape());
	}

}
