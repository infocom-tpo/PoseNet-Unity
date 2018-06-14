using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


public partial class PoseNet
{
    const int kLocalMaximumRadius = 1;
    public int NUM_KEYPOINTS = 0;
    public String[] partNames;
    public Dictionary<String, int> partIds;
    public Tuple<string, string>[] connectedPartNames;
    public Tuple<int, int>[] connectedPartIndices;
    public Tuple<string, string>[] poseChain;
    public Tuple<int, int>[] parentChildrenTuples;
    public int[] parentToChildEdges;
    public int[] childToParentEdges;

    public PoseNet()
    {
        partNames = new String[]{
            "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
            "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
            "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
        };

        NUM_KEYPOINTS = partNames.Length;

        partIds = partNames
            .Select((k, v) => new { k, v })
            .ToDictionary(p => p.k, p => p.v);

        connectedPartNames = new Tuple<string, string>[] {
            Tuple.Create("leftHip", "leftShoulder"), Tuple.Create("leftElbow", "leftShoulder"),
            Tuple.Create("leftElbow", "leftWrist"), Tuple.Create("leftHip", "leftKnee"),
            Tuple.Create("leftKnee", "leftAnkle"), Tuple.Create("rightHip", "rightShoulder"),
            Tuple.Create("rightElbow", "rightShoulder"), Tuple.Create("rightElbow", "rightWrist"),
            Tuple.Create("rightHip", "rightKnee"), Tuple.Create("rightKnee", "rightAnkle"),
            Tuple.Create("leftShoulder", "rightShoulder"), Tuple.Create("leftHip", "rightHip")
        };

        connectedPartIndices = connectedPartNames.Select(x =>
          new Tuple<int, int>(partIds[x.Item1], partIds[x.Item2])
        ).ToArray();

        poseChain = new Tuple<string, string>[]{
            Tuple.Create("nose", "leftEye"), Tuple.Create("leftEye", "leftEar"), Tuple.Create("nose", "rightEye"),
            Tuple.Create("rightEye", "rightEar"), Tuple.Create("nose", "leftShoulder"),
            Tuple.Create("leftShoulder", "leftElbow"), Tuple.Create("leftElbow", "leftWrist"),
            Tuple.Create("leftShoulder", "leftHip"), Tuple.Create("leftHip", "leftKnee"),
            Tuple.Create("leftKnee", "leftAnkle"), Tuple.Create("nose", "rightShoulder"),
            Tuple.Create("rightShoulder", "rightElbow"), Tuple.Create("rightElbow", "rightWrist"),
            Tuple.Create("rightShoulder", "rightHip"), Tuple.Create("rightHip", "rightKnee"),
            Tuple.Create("rightKnee", "rightAnkle")
        };

        parentChildrenTuples = poseChain.Select(x =>
          new Tuple<int, int>(partIds[x.Item1], partIds[x.Item2])
        ).ToArray();

        parentToChildEdges = parentChildrenTuples.Select(x => x.Item2).ToArray();
        childToParentEdges = parentChildrenTuples.Select(x => x.Item1).ToArray();
    }
}