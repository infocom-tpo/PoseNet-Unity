using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using TensorFlow;
using System;

public class ReadImageExample : MonoBehaviour {
	int ImageSize= 512;

    PoseNet posenet = new PoseNet();
    PoseNet.Pose[] poses;

    public GameObject glgo;
    private GLRenderer gl;

    void Start () {

		TextAsset graphModel = Resources.Load ("frozen_model") as TextAsset;
		var graph = new TFGraph ();
		graph.Import (graphModel.bytes);
		var session = new TFSession (graph);

        Texture2D image = Resources.Load("tennis_in_crowd") as Texture2D;
        image = scaled(image, ImageSize, ImageSize);
        var tensor = TransformInput(image.GetPixels32(), ImageSize, ImageSize);

        var runner = session.GetRunner ();
		runner.AddInput (graph ["image"] [0], tensor);
		runner.Fetch(
			graph ["heatmap"] [0],
			graph ["offset_2"] [0],
			graph ["displacement_fwd_2"] [0],
			graph ["displacement_bwd_2"] [0]
		);
		var result = runner.Run();
		var heatmap = (float [,,,])result [0].GetValue (jagged: false);
		var offsets = (float [,,,])result [1].GetValue (jagged: false);
		var displacementsFwd = (float [,,,])result [2].GetValue (jagged: false);
		var displacementsBwd = (float [,,,])result [3].GetValue (jagged: false);

		//Debug.Log(mean(heatmap));

        poses = posenet.DecodeMultiplePoses(
            heatmap, offsets,
            displacementsFwd,
            displacementsBwd,
            outputStride: 16, maxPoseDetections: 15,
            scoreThreshold: 0.5f, nmsRadius: 20);

        gl = glgo.GetComponent<GLRenderer>();
    }

    public void OnRenderObject()
    {
        gl.DrawResults(poses);
    }
    
    // Update is called once per frame
    void Update () {
		
	}

    public static TFTensor TransformInput(Color32[] pic, int width, int height)
	{
		System.Array.Reverse(pic);
		float[] floatValues = new float[width * height * 3];
		//double sum = 0f;

		for (int i = 0; i < pic.Length; ++i)
		{
			var color = pic[i];
			floatValues [i * 3 + 0] = color.r * (2.0f / 255.0f) - 1.0f;
			floatValues [i * 3 + 1] = color.g * (2.0f / 255.0f) - 1.0f;
			floatValues [i * 3 + 2] = color.b * (2.0f / 255.0f) - 1.0f;

		}

		TFShape shape = new TFShape(1, width, height, 3);

		return TFTensor.FromBuffer(shape, floatValues, 0, floatValues.Length);
	}

	public static Texture2D scaled(Texture2D src, int width, int height, FilterMode mode = FilterMode.Trilinear)
	{
			Rect texR = new Rect(0,0,width,height);
			_gpu_scale(src,width,height,mode);
			
			//Get rendered data back to a new texture
			Texture2D result = new Texture2D(width, height, TextureFormat.ARGB32, true);
			result.Resize(width, height);
			result.ReadPixels(texR,0,0,true);
			return result;                 
	}
	static void _gpu_scale(Texture2D src, int width, int height, FilterMode fmode)
	{
			//We need the source texture in VRAM because we render with it
			src.filterMode = fmode;
			src.Apply(true);       
							
			//Using RTT for best quality and performance. Thanks, Unity 5
			RenderTexture rtt = new RenderTexture(width, height, 32);
			
			//Set the RTT in order to render to it
			Graphics.SetRenderTarget(rtt);
			
			//Setup 2D matrix in range 0..1, so nobody needs to care about sized
			GL.LoadPixelMatrix(0,1,1,0);
			
			//Then clear & draw the texture to fill the entire RTT.
			GL.Clear(true,true,new Color(0,0,0,0));
			Graphics.DrawTexture(new Rect(0,0,1,1),src);
    }
}
