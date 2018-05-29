using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;

public class WebCameraExample : MonoBehaviour {

    public int Width = 512;
    public int Height = 512;
    public int FPS = 30;
    WebCamTexture webcamTexture;
    GLRenderer gl;
    int ImageSize = 512;
    PoseNet posenet = new PoseNet();
    PoseNet.Pose[] poses;
    TFSession session;
    //TFSession.Runner runner;
    TFGraph graph;
    bool isPosing;

    // Use this for initialization
    void Start () {
        WebCamDevice[] devices = WebCamTexture.devices;
        webcamTexture = new WebCamTexture(devices[0].name, Width, Height, FPS);
        GetComponent<Renderer>().material.mainTexture = webcamTexture;
        webcamTexture.Play();

        TextAsset graphModel = Resources.Load("frozen_model") as TextAsset;
        graph = new TFGraph();
        graph.Import(graphModel.bytes);
        session = new TFSession(graph);
        
        gl = GameObject.Find("GLRender").GetComponent<GLRenderer>();

    }
	
	// Update is called once per frame
	void Update () {
        var color32 = webcamTexture.GetPixels32();

        Texture2D texture = new Texture2D(webcamTexture.width, webcamTexture.height);

        texture.SetPixels32(color32);
        texture.Apply();

        if (isPosing) return;
        isPosing = true;
        StartCoroutine("PoseUpdate", texture);
        texture = null;

    }

    IEnumerator PoseUpdate(Texture2D texture)
    {

        texture = scaled(texture, ImageSize, ImageSize);
        var tensor = TransformInput(texture.GetPixels32(), ImageSize, ImageSize);

        var runner = session.GetRunner();
        runner.AddInput(graph["image"][0], tensor);
        runner.Fetch(
            graph["heatmap"][0],
            graph["offset_2"][0],
            graph["displacement_fwd_2"][0],
            graph["displacement_bwd_2"][0]
        );

        var result = runner.Run();
        var heatmap = (float[,,,])result[0].GetValue(jagged: false);
        var offsets = (float[,,,])result[1].GetValue(jagged: false);
        var displacementsFwd = (float[,,,])result[2].GetValue(jagged: false);
        var displacementsBwd = (float[,,,])result[3].GetValue(jagged: false);

       // Debug.Log(PoseNet.mean(heatmap));

        poses = posenet.DecodeMultiplePoses(
            heatmap, offsets,
            displacementsFwd,
            displacementsBwd,
            outputStride: 16, maxPoseDetections: 15,
            scoreThreshold: 0.5f, nmsRadius: 20);

        isPosing = false;

        texture = null;
        Resources.UnloadUnusedAssets();


        yield return null;
    }

    public void OnRenderObject()
    {
        //Debug.Log(poses);
        gl.DrawResults(poses);
    }

    public static TFTensor TransformInput(Color32[] pic, int width, int height)
    {
        System.Array.Reverse(pic);
        float[] floatValues = new float[width * height * 3];
        //double sum = 0f;

        for (int i = 0; i < pic.Length; ++i)
        {
            var color = pic[i];
            floatValues[i * 3 + 0] = color.r * (2.0f / 255.0f) - 1.0f;
            floatValues[i * 3 + 1] = color.g * (2.0f / 255.0f) - 1.0f;
            floatValues[i * 3 + 2] = color.b * (2.0f / 255.0f) - 1.0f;

        }

        TFShape shape = new TFShape(1, width, height, 3);

        return TFTensor.FromBuffer(shape, floatValues, 0, floatValues.Length);
    }

    public static Texture2D scaled(Texture2D src, int width, int height, FilterMode mode = FilterMode.Trilinear)
    {
        Rect texR = new Rect(0, 0, width, height);
        _gpu_scale(src, width, height, mode);

        //Get rendered data back to a new texture
        Texture2D result = new Texture2D(width, height, TextureFormat.ARGB32, true);
        result.Resize(width, height);
        result.ReadPixels(texR, 0, 0, true);
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
        GL.LoadPixelMatrix(0, 1, 1, 0);

        //Then clear & draw the texture to fill the entire RTT.
        GL.Clear(true, true, new Color(0, 0, 0, 0));
        Graphics.DrawTexture(new Rect(0, 0, 1, 1), src);
    }
}
