using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using TensorFlow;
using System;

public class ReadImage : MonoBehaviour {
	int ImageSize= 512;
    //Renderer m_ObjectRenderer;
    // Use this for initialization

    PoseNet posenet = new PoseNet();
    PoseNet.Pose[] poses;

    void Awake()
    {

    }
    void Start () {

		TextAsset graphModel = Resources.Load ("frozen_model") as TextAsset;
		var graph = new TFGraph ();
		graph.Import (graphModel.bytes);
		var session = new TFSession (graph);

        //Texture2D image = Resources.Load ("tennis_in_crowd") as Texture2D;

        Texture2D image = Resources.Load("frisbee") as Texture2D;
        image = scaled(image, ImageSize, ImageSize);
        var tensor = TransformInput(image.GetPixels32(), ImageSize, ImageSize);

        //TextAsset image2 = Resources.Load("image") as TextAsset;
        //var img = GetByteToFloat(image2.bytes);
        //var tensor = TransformInput(img, ImageSize, ImageSize);

        //Debug.Log(img[0]);

        var runner = session.GetRunner ();
		runner.AddInput (graph ["image"] [0], tensor);
		runner.Fetch(
			graph ["heatmap"] [0],
			graph ["offset_2"] [0],
			graph ["displacement_fwd_2"] [0],
			graph ["displacement_bwd_2"] [0]
		);
		var result = runner.Run();
		//Debug.Log(result[0]);
		//Debug.Log(result[0].Data);
		//Debug.Log(result[0].TensorByteSize);
		var heatmap = (float [,,,])result [0].GetValue (jagged: false);
		var offsets = (float [,,,])result [1].GetValue (jagged: false);
		var displacementsFwd = (float [,,,])result [2].GetValue (jagged: false);
		var displacementsBwd = (float [,,,])result [3].GetValue (jagged: false);

		Debug.Log(mean(heatmap));

        poses = posenet.DecodeMultiplePoses(
            heatmap, offsets,
            displacementsFwd,
            displacementsBwd,
            outputStride: 16, maxPoseDetections: 15,
            scoreThreshold: 0.5f, nmsRadius: 20);
    }

    public void OnRenderObject()
    {
        DrawResults(poses);
    }

    static Material lineMaterial;

    static void CreateLineMaterial()
    {
        if (!lineMaterial)
        {
            Shader shader = Shader.Find("Hidden/Internal-Colored");
            lineMaterial = new Material(shader);
            lineMaterial.hideFlags = HideFlags.HideAndDontSave;
            lineMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            lineMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            lineMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            lineMaterial.SetInt("_ZWrite", 0);
        }
    }

    public void DrawResults(PoseNet.Pose[] poses)
    {
        CreateLineMaterial();
        lineMaterial.SetPass(0);
        GL.PushMatrix();
        GL.MultMatrix(transform.localToWorldMatrix);
        GL.Begin(GL.LINES);
        GL.Color(Color.red);
        float minPoseConfidence = 0.5f;


        foreach (var pose in poses)
        {
            if (pose.score >= minPoseConfidence)
            {
                DrawSkeleton(pose.keypoints,
                    minPoseConfidence, 0.02f);
            }
        }

        GL.End();
        GL.PopMatrix();
    }

    public void DrawSkeleton(PoseNet.Keypoint[] keypoints, float minConfidence, float scale)
    {

        var adjacentKeyPoints = GetAdjacentKeyPoints(
            keypoints, minConfidence);


        foreach (var keypoint in adjacentKeyPoints)
        {
            GL.Vertex3(keypoint.Item1.position.x * scale, keypoint.Item1.position.y * scale, 0);
            GL.Vertex3(keypoint.Item2.position.x * scale, keypoint.Item2.position.y * scale, 0);
        }
    }

    bool EitherPointDoesntMeetConfidence(
        float a, float b, float minConfidence) {
        return (a < minConfidence || b < minConfidence);
    }

    Tuple<PoseNet.Keypoint, PoseNet.Keypoint>[] GetAdjacentKeyPoints(
        PoseNet.Keypoint[] keypoints, float minConfidence) {

        return posenet.connectedPartIndeces.Where(x =>
           !EitherPointDoesntMeetConfidence(keypoints[x.Item1].score, keypoints[x.Item2].score, minConfidence))
           .Select(x => new Tuple<PoseNet.Keypoint, PoseNet.Keypoint>(keypoints[x.Item1], keypoints[x.Item2])).ToArray();

    }

    public static double mean(float[,,,] tensor)
    {
        double sum = 0f;
        var x = tensor.GetLength(1);
        var y = tensor.GetLength(2);
        var z = tensor.GetLength(3);
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                for (int k = 0; k < z; k++)
                {
                    sum += tensor[0, i, j, k];
                }
            }
        }
        var mean = sum / (x * y * z);
        return mean;
    }

    public static float[] GetByteToFloat(byte[] byteArray)
    {
        var floatArray = new float[byteArray.Length / 4];
        System.Buffer.BlockCopy(byteArray, 0, floatArray, 0, byteArray.Length);
        return floatArray;
    }

    // public static float[] GetByteToFloat(byte[] byteArray) {
    // 	var floatArray = new float[byteArray.Length / 4];
    // 	System.Buffer.BlockCopy(byteArray, 0, floatArray, 0, byteArray.Length);
    // 	return floatArray;
    // }

    // Update is called once per frame
    void Update () {
		
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

			//sum += floatValues[i * 3 + 0] + floatValues [i * 3 + 1] + floatValues [i * 3 + 2];
		}

		//Debug.Log(sum/ pic.Length / 3);

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
