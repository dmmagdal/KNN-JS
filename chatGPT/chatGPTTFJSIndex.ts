import * as tf from '@tensorflow/tfjs';

class VectorIndex {
  private vectors: tf.Tensor2D;

  constructor() {
    this.vectors = tf.tensor2d([], [0, 768]); // Initialize an empty tensor
  }

  public add(vector: tf.Tensor1D): void {
    this.vectors = tf.concat([this.vectors, vector.reshape([1, 768])], 0);
  }

  public delete(targetVector: tf.Tensor1D): void {
    const index = this.findVectorIndex(targetVector);
    if (index >= 0) {
      this.vectors = tf.concat([this.vectors.slice([0, 0], [index, -1]), this.vectors.slice([index + 1, 0], [-1, -1])], 0);
    }
  }

  public async get(queryVector: tf.Tensor1D, k: number): Promise<tf.Tensor2D> {
    const distances = tf.norm(this.vectors.sub(queryVector), 2, 1); // Calculate L2 (Euclidean) distances
    const { values, indices } = tf.topk(distances, k, true); // Get indices and values of k nearest neighbors

    const nearestNeighbors = this.vectors.gather(indices); // Gather the nearest neighbors from the vectors

    return nearestNeighbors;
  }

  private findVectorIndex(targetVector: tf.Tensor1D): number {
    const isEqual = this.vectors.equal(targetVector.reshape([1, -1])).all(axis = 1);
    const indexTensor = isEqual.argMax();

    const index = indexTensor.arraySync()[0];

    if (index !== null && index !== undefined) {
      return index;
    }

    return -1;
  }
}
