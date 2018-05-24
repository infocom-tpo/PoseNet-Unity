using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;

public class Sample : MonoBehaviour {

	int ImageSize= 512;
	// Use this for initialization
	void Start () {
		TextAsset graphModel = Resources.Load ("frozen_model") as TextAsset;
		var graph = new TFGraph ();
		graph.Import (graphModel.bytes);
		var session = new TFSession (graph);

		TextAsset image = Resources.Load ("image") as TextAsset;
		var img = GetByteToFloat(image.bytes);

		var tensor = TransformInput(img,ImageSize,ImageSize);

		var runner = session.GetRunner ();
		runner.AddInput (graph ["image"] [0], tensor);
		runner.Fetch(
			graph ["heatmap"] [0],
			graph ["offset_2"] [0],
			graph ["displacement_fwd_2"] [0],
			graph ["displacement_bwd_2"] [0]
		);
		var result = runner.Run();
		Debug.Log(result[0]);
		Debug.Log(result[0].Data);
		Debug.Log(result[0].TensorByteSize);
		var heatmap = (float [,,,])result [0].GetValue (jagged: false);

		Debug.Log(mean(heatmap));
	}
	
	// Update is called once per frame
	void Update () {
		
	}

	public static double mean(float[,,,] tensor){
		double sum = 0f;
		var x = tensor.GetLength (1);
		var y = tensor.GetLength (2);
		var z = tensor.GetLength (3);
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				for (int k = 0; k < z; k++) {
					sum += tensor[0,i,j,k];
				}
			}
		}
		var mean = sum / (x * y * z);
		return mean;
	}

	public static TFTensor TransformInput(float[] floatValues, int width, int height)
	{

		// float[] floatValues = new float[pic.Length];
		double sum = 0f;
		for (int i = 0; i < floatValues.Length; ++i)
		{
			sum += floatValues[i];
		}
		Debug.Log(sum / floatValues.Length);
		
		TFShape shape = new TFShape(1, width, height, 3);

		return TFTensor.FromBuffer(shape, floatValues, 0, floatValues.Length);
	}
	public static float[] GetByteToFloat(byte[] byteArray) {
		var floatArray = new float[byteArray.Length / 4];
		System.Buffer.BlockCopy(byteArray, 0, floatArray, 0, byteArray.Length);
		return floatArray;
	}
}
