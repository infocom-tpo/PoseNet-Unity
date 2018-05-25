using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Linq;
using TensorFlow;

public class ByteArrayTest : MonoBehaviour {
    PoseNet posenet = new PoseNet();
    PoseNet.Pose[] poses;

    // Use this for initialization
    void Start () {
		
		var heatmap = GetResourcesToFloats("heatmaps",33,33,17);
		var offsets = GetResourcesToFloats("offsets",33,33,34);
		var displacementsFwd = GetResourcesToFloats("displacementsFwd",33,33,32);
		var displacementsBwd = GetResourcesToFloats("displacementsBwd",33,33,32);

        poses = posenet.DecodeMultiplePoses(
            heatmap, offsets,
            displacementsFwd,
            displacementsBwd,
            outputStride: 16, maxPoseDetections: 15,
            scoreThreshold: 0.5f, nmsRadius: 20);
    }

	public static float[,,,] GetResourcesToFloats(string name, int y,int x,int z) {
		TextAsset binary = Resources.Load (name) as TextAsset;
		var floatValues = GetByteToFloat(binary.bytes);
		var tensor = GetFloatToTensor(floatValues, y, x, z);
		return (float[,,,])tensor.GetValue();
	}

	public static TFTensor GetFloatToTensor(float[] floatValues, int y,int x,int z) {
		TFShape shape = new TFShape(1, y, x, z);
		var tensor = TFTensor.FromBuffer(shape, floatValues, 0, floatValues.Length);
		return tensor;
	}

	public static float[] GetByteToFloat(byte[] byteArray) {
		var floatArray = new float[byteArray.Length / 4];
		Buffer.BlockCopy(byteArray, 0, floatArray, 0, byteArray.Length);
		return floatArray;
	}
	
	// Update is called once per frame
	void Update () {

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
            //DrawResults(poses);
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
    Tuple<PoseNet.Keypoint, PoseNet.Keypoint>[] GetAdjacentKeyPoints(
            PoseNet.Keypoint[] keypoints, float minConfidence)
    {

        return posenet.connectedPartIndeces
            .Where(x => !EitherPointDoesntMeetConfidence(
                keypoints[x.Item1].score, keypoints[x.Item2].score, minConfidence))
           .Select(x => new Tuple<PoseNet.Keypoint, PoseNet.Keypoint>(keypoints[x.Item1], keypoints[x.Item2])).ToArray();

    }
    bool EitherPointDoesntMeetConfidence(
        float a, float b, float minConfidence)
    {
        return (a < minConfidence || b < minConfidence);
    }

}
