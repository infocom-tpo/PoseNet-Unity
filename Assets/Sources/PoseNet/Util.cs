using UnityEngine;
using System.Linq;
using System;

public partial class PoseNet
{

    Vector2 GetOffsetPoint(int y,int x,int keypoint, float[,,,] offsets) {
        return new Vector2(
            offsets[0, y, x, keypoint + NUM_KEYPOINTS],
            offsets[0, y, x, keypoint]
        );
    }

    float SquaredDistance(
        float y1, float x1, float y2, float x2) {
        var dy = y2 - y1;
        var dx = x2 - x1;
        return dy * dy + dx * dx;
    }

    Vector2 AddVectors(Vector2 a, Vector2 b) {
        return new Vector2(x: a.x + b.x, y: a.y + b.y);
    }

    Vector2 GetImageCoords(
        Part part,int outputStride,float[,,,] offsets) {
        var vec = GetOffsetPoint(part.heatmapY, part.heatmapX,
                                 part.id, offsets);
        return new Vector2(
            (float)(part.heatmapX * outputStride) + vec.x,
            (float)(part.heatmapY * outputStride) + vec.y
        );
    }

    public Tuple<Keypoint, Keypoint>[] GetAdjacentKeyPoints(
           Keypoint[] keypoints, float minConfidence)
    {

        return connectedPartIndices
            .Where(x => !EitherPointDoesntMeetConfidence(
                keypoints[x.Item1].score, keypoints[x.Item2].score, minConfidence))
           .Select(x => new Tuple<Keypoint, Keypoint>(keypoints[x.Item1], keypoints[x.Item2])).ToArray();

    }

    bool EitherPointDoesntMeetConfidence(
        float a, float b, float minConfidence)
    {
        return (a < minConfidence || b < minConfidence);
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

    //Pose ScalePose(Pose pose, int scale) {

    //    var s = (float)scale;

    //    return new Pose(
    //        pose.keypoints.Select( x => 
    //            new Keypoint( 
    //                x.score,
    //                new Vector2(x.position.x * s, x.position.y * s),
    //                x.part)
    //         ).ToArray(),
    //         pose.score
    //     );
    //}

    //Pose[] ScalePoses(Pose[] poses, int scale) {
    //    if (scale == 1) {
    //        return poses;
    //    }
    //    return poses.Select(x => ScalePose(pose: x, scale: scale)).ToArray();
    //}

    //int GetValidResolution(float imageScaleFactor,
    //                       int inputDimension,
    //                       int outputStride) {
    //    var evenResolution = (int)(inputDimension * imageScaleFactor) - 1;
    //    return evenResolution - (evenResolution % outputStride) + 1;
    //}

    //int Half(int k)
    //{
    //    return (int)Mathf.Floor((float)(k / 2));
    //}


}