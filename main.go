package main

import (
  "fmt"
  "math"
  "math/rand"
)

var (
  inputs          []float64
  hiddenBias      []float64
  hiddenWeights   [][]float64
  outputBias      float64
  outputWeights   []float64
  learningRate    float64
  target, predicted float64
  hiddenLayerOutput []float64
)

func makeNN(numInputs, numHidden int) {
  inputs = make([]float64, numInputs)
  hiddenBias = make([]float64, numHidden)
  hiddenWeights = make([][]float64, numHidden)
  outputWeights = make([]float64, numHidden)
  learningRate = 0.0007

  for i := range hiddenBias {
    hiddenBias[i] = rand.Float64()
    hiddenWeights[i] = make([]float64, numInputs)
    for j := range hiddenWeights[i] {
      hiddenWeights[i][j] = rand.Float64()
    }
  }

  outputBias = rand.Float64()
  for i := range outputWeights {
    outputWeights[i] = rand.Float64()
  }

  for i := range inputs {
    inputs[i] = rand.Float64() * 9 + 1
  }
}

func relu(x float64) float64 {
  if x > 0 {
    return x
  }
  return 0
}

func reluDerivative(x float64) float64 {
  if x > 0 {
    return 1
  }
  return 0
}

func cost() float64 {
  return target - predicted
}

func backprop() {
  outputGradient := cost() * reluDerivative(predicted)

  for i := range outputWeights {
    outputWeights[i] += learningRate * outputGradient * hiddenLayerOutput[i]
  }
  outputBias += learningRate * outputGradient

  hiddenGradients := make([]float64, len(hiddenWeights))
  for i := range hiddenWeights {
    hiddenGradients[i] = outputGradient * outputWeights[i] * reluDerivative(hiddenLayerOutput[i])
  }

  for i := range hiddenWeights {
    for j := range hiddenWeights[i] {
      hiddenWeights[i][j] += learningRate * hiddenGradients[i] * inputs[j]
    }
    hiddenBias[i] += learningRate * hiddenGradients[i]
  }
}

func feedForward() float64 {
  hiddenLayerOutput = make([]float64, len(hiddenWeights))

  for i := range hiddenWeights {
    hiddenInput := 0.0
    for j := range inputs {
      hiddenInput += inputs[j] * hiddenWeights[i][j]
    }
    hiddenInput += hiddenBias[i]
    hiddenLayerOutput[i] = relu(hiddenInput)
  }

  outputInput := 0.0
  for i := range hiddenLayerOutput {
    outputInput += hiddenLayerOutput[i] * outputWeights[i]
  }
  outputInput += outputBias
  predicted = relu(outputInput)

  return predicted
}

func main() {
  makeNN(2, 3)

  for {
    target = inputs[0] + inputs[1]
    feedForward()
    backprop()
    if math.Abs(cost()) <= 0.0001 {
      inputs[0] = rand.Float64() * 9 + 1
      inputs[1] = rand.Float64() * 9 + 1
      target = inputs[0] + inputs[1]
      feedForward()
      if math.Abs(cost()) <= 0.0001 {
        fmt.Println("Finished without learning")
      }
    }
  }
}
