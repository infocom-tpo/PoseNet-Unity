using UnityEngine;
using UnityEditor;
using TensorFlow;
using System.Collections.Generic;
using System.Linq;

public partial class PoseNet {

    bool WithinNmsRadiusOfCorrespondingPoint(
        List<Pose> poses, float squaredNmsRadius, Vector2 vec, int keypointId)
    {
        return poses.Any(pose =>
            SquaredDistance(vec.y, vec.x,
                pose.keypoints[keypointId].position.y,
                pose.keypoints[keypointId].position.x) <= squaredNmsRadius
        );
        //foreach (var pose in poses)
        //{
        //    if (SquaredDistance(vec.y, vec.x,
        //        pose.keypoints[keypointId].position.y,
        //        pose.keypoints[keypointId].position.x) <= squaredNmsRadius)
        //    {
        //        return true;
        //    }
        //}
        //return false;
    }
    
    float GetInstanceScore(
        List<Pose> existingPoses, float squaredNmsRadius,
        Keypoint[] instanceKeypoints) {

        float notOverlappedKeypointScores = instanceKeypoints
           .Where((x, id) => !WithinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius, x.position, id))
           .Sum(x => x.score);

        //int id = 0;
        //float notOverlappedKeypointScores = 0.0f;
        //foreach (var x in instanceKeypoints)
        //{
        //    if (!WithinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius, x.position, id))
        //    {
        //        notOverlappedKeypointScores += x.score;
        //    }
        //}
        return notOverlappedKeypointScores / instanceKeypoints.Length;

    }

    public Pose[] DecodeMultiplePoses(
        float[,,,] scores, float[,,,] offsets,
        float[,,,] displacementsFwd, float[,,,] displacementBwd,
        int outputStride, int maxPoseDetections,
        float scoreThreshold, int nmsRadius = 20)
    {
        var poses = new List<Pose>();
        var squaredNmsRadius = (float)nmsRadius * nmsRadius;

        PriorityQueue<float, PartWithScore> queue = BuildPartWithScoreQueue(
            scoreThreshold, kLocalMaximumRadius, scores);

        while (poses.Count < maxPoseDetections && queue.Count > 0)
        {            
            var root = queue.Pop().Value;

            // Part-based non-maximum suppression: We reject a root candidate if it
            // is within a disk of `nmsRadius` pixels from the corresponding part of
            // a previously detected instance.
            var rootImageCoords =
                GetImageCoords(root.part, outputStride, offsets);

            if (WithinNmsRadiusOfCorrespondingPoint(
                    poses, squaredNmsRadius, rootImageCoords, root.part.id))
            {
                continue;
            }

            // Start a new detection instance at the position of the root.
            var keypoints = DecodePose(
                root, scores, offsets, outputStride, displacementsFwd,
                displacementBwd);

            var score = GetInstanceScore(poses, squaredNmsRadius, keypoints);
            poses.Add(new Pose(keypoints, score ));
        }

        return poses.ToArray();
    }
}