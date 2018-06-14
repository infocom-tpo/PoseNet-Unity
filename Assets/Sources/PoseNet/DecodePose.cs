using UnityEngine;

public partial class PoseNet
{

    Vector2 GetDisplacement(int edgeId, Vector2Int point, float[,,,] displacements) {

        var numEdges = (int)(displacements.GetLength(3) / 2);

        return new Vector2(
            displacements[0, point.y, point.x, numEdges + edgeId],
            displacements[0, point.y, point.x, edgeId]
        );
    }

    Vector2Int GetStridedIndexNearPoint(
        Vector2 point, int outputStride, int height,
        int width) {

        return new Vector2Int(
            (int)Mathf.Clamp(Mathf.Round(point.x / outputStride), 0, width - 1),
            (int)Mathf.Clamp(Mathf.Round(point.y / outputStride), 0, height - 1)
        );
    }

    /**
     * We get a new keypoint along the `edgeId` for the pose instance, assuming
     * that the position of the `idSource` part is already known. For this, we
     * follow the displacement vector from the source to target part (stored in
     * the `i`-t channel of the displacement tensor).
     */

    Keypoint TraverseToTargetKeypoint(
        int edgeId, Keypoint sourceKeypoint, int targetKeypointId,
        float[,,,] scores, float[,,,] offsets, int outputStride,
        float[,,,] displacements) {

        var height = scores.GetLength(1);
        var width = scores.GetLength(2);

        // Nearest neighbor interpolation for the source->target displacements.
        var sourceKeypointIndices = GetStridedIndexNearPoint(
            sourceKeypoint.position, outputStride, height, width);

        var displacement =
            GetDisplacement(edgeId, sourceKeypointIndices, displacements);

        var displacedPoint = AddVectors(sourceKeypoint.position, displacement);

        var displacedPointIndices =
            GetStridedIndexNearPoint(displacedPoint, outputStride, height, width);

        var offsetPoint = GetOffsetPoint(
                displacedPointIndices.y, displacedPointIndices.x, targetKeypointId,
                offsets);

        var score = scores[0,
            displacedPointIndices.y, displacedPointIndices.x, targetKeypointId];

        var targetKeypoint =
            AddVectors(
                new Vector2(
                    x: displacedPointIndices.x * outputStride,
                    y: displacedPointIndices.y * outputStride)
                , new Vector2(x: offsetPoint.x, y: offsetPoint.y));

        return new Keypoint(score, targetKeypoint, partNames[targetKeypointId]);
    }

    Keypoint[] DecodePose(PartWithScore root, float[,,,] scores, float[,,,] offsets,
        int outputStride, float[,,,] displacementsFwd,
        float[,,,] displacementsBwd)
    {

        var numParts = scores.GetLength(3);
        var numEdges = parentToChildEdges.Length;

        var instanceKeypoints = new Keypoint[numParts];

        // Start a new detection instance at the position of the root.
        var rootPart = root.part;
        var rootScore = root.score;
        var rootPoint = GetImageCoords(rootPart, outputStride, offsets);

        instanceKeypoints[rootPart.id] = new Keypoint(
            rootScore,
            rootPoint,
            partNames[rootPart.id]
        );

        // Decode the part positions upwards in the tree, following the backward
        // displacements.
        for (var edge = numEdges - 1; edge >= 0; --edge)
        {
            var sourceKeypointId = parentToChildEdges[edge];
            var targetKeypointId = childToParentEdges[edge];
            if (instanceKeypoints[sourceKeypointId].score > 0.0f &&
                instanceKeypoints[targetKeypointId].score == 0.0f)
            {
                instanceKeypoints[targetKeypointId] = TraverseToTargetKeypoint(
                    edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
                    offsets, outputStride, displacementsBwd);
            }
        }

        // Decode the part positions downwards in the tree, following the forward
        // displacements.
        for (var edge = 0; edge < numEdges; ++edge)
        {
            var sourceKeypointId = childToParentEdges[edge];
            var targetKeypointId = parentToChildEdges[edge];
            if (instanceKeypoints[sourceKeypointId].score > 0.0f &&
                instanceKeypoints[targetKeypointId].score == 0.0f)
            {
                instanceKeypoints[targetKeypointId] = TraverseToTargetKeypoint(
                    edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
                    offsets, outputStride, displacementsFwd);
            }
        }

        return instanceKeypoints;

    }
}