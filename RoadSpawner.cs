using UnityEngine;
using System.Collections.Generic;

public class RoadSpawner : MonoBehaviour
{
    public GameObject roadPrefab;
    public Transform player;
    public int numberOfSegments = 5;
    public float segmentLength = 10f;
    public float spawnThreshold = 15f;

    private List<GameObject> spawnedSegments = new List<GameObject>();
    private float lastSpawnZ;

    void Start()
    {
        lastSpawnZ = -segmentLength; // Para que el primer spawn sea justo en Z=0

        for (int i = 0; i < numberOfSegments; i++)
        {
            SpawnSegment();
        }
    }

    void Update()
    {
        if (player.position.z > lastSpawnZ - (numberOfSegments * segmentLength - spawnThreshold))
        {
            SpawnSegment();
            DestroyOldestSegment();
        }
    }

    void SpawnSegment()
    {
        lastSpawnZ += segmentLength;
        GameObject segment = Instantiate(roadPrefab, new Vector3(0, 0, lastSpawnZ), Quaternion.identity);
        spawnedSegments.Add(segment);
    }

    void DestroyOldestSegment()
    {
        if (spawnedSegments.Count > numberOfSegments)
        {
            Destroy(spawnedSegments[0]);
            spawnedSegments.RemoveAt(0);
        }
    }
}

