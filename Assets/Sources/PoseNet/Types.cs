using UnityEngine;

public partial class PoseNet
{
    public struct PartWithScore
    {
        public float score;
        public Part part;

        public PartWithScore(float score,Part part)
        {
            this.score = score;
            this.part = part;
        }

    }
    public struct Part
    {
        public int heatmapX;
        public int heatmapY;
        public int id;
        public Part(int heatmapX, int heatmapY, int id)
        {
            this.heatmapX = heatmapX;
            this.heatmapY = heatmapY;
            this.id = id;
        }
    }

    public struct Keypoint
    {
        public float score;
        public Vector2 position;
        public string part;

        public Keypoint(float score, Vector2 position, string part)
        {
            this.score = score;
            this.position = position;
            this.part = part;
        }
    }

    public struct Pose
    {
        public Keypoint[] keypoints;
        public float score;

        public Pose(Keypoint[] keypoints, float score)
        {
            this.keypoints = keypoints;
            this.score = score;
        }
    }

}


