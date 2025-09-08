package main

import (
	//No third-party libraries
	"fmt"
	"math"
	"math/rand"
)

var (
	weights           [][][]float64
	biases            [][]float64
	layerOutputs      [][]float64
	learningRate      float64
	inputs            []float64
	target, predicted float64
	dropoutProb       float64
)

// Initializes the Neural Network
func makeNN(layerSizes []int) {
	numLayers := len(layerSizes)
	weights = make([][][]float64, numLayers-1)
	biases = make([][]float64, numLayers-1)
	layerOutputs = make([][]float64, numLayers)

	for i := range weights {
		weights[i] = make([][]float64, layerSizes[i+1])
		for j := range weights[i] {
			weights[i][j] = make([]float64, layerSizes[i])
			for k := range weights[i][j] {
				weights[i][j][k] = rand.Float64()
			}
		}
	}

	for i := range biases {
		biases[i] = make([]float64, layerSizes[i+1])
		for j := range biases[i] {
			biases[i][j] = rand.Float64()
		}
	}

	for i := range layerOutputs {
		layerOutputs[i] = make([]float64, layerSizes[i])
	}

	inputs = make([]float64, layerSizes[0])
	learningRate = 0.0007

	for i := range inputs {
		inputs[i] = rand.Float64()*9 + 1
	}
}

// Activation functions
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

func sigmoid(x float64) float64 {
	return 1 - (1 / (math.Exp(-x)))
}

func sigmoidDerivative(x float64) float64 {
	return sigmoid(x) * (-sigmoid(x))
}

func tanh(x float64) float64 {
	return (2 / (1 + (math.Exp(-(2 * x))))) - 1
}

func tanhDerivative(x float64) float64 {
	t := tanh(x)
	return 1 - t*t
}

func softplus(x float64) float64 {
	return math.Log(1 + math.Exp(x))
}

func softplusDerivative(x float64) float64 {
	return sigmoid(x)
}

func cost() float64 {
	return target - predicted
}

// Backpropogation
func backprop() {
	numLayers := len(weights) + 1

	outputGradient := cost() * reluDerivative(predicted)
	delta := make([][]float64, numLayers)
	delta[numLayers-1] = make([]float64, len(layerOutputs[numLayers-1]))
	for i := range delta[numLayers-1] {
		delta[numLayers-1][i] = outputGradient
	}

	for l := numLayers - 2; l >= 0; l-- {
		delta[l] = make([]float64, len(layerOutputs[l]))
		for i := range delta[l] {
			error := 0.0
			for j := range weights[l] {
				error += delta[l+1][j] * weights[l][j][i]
			}
			delta[l][i] = error * reluDerivative(layerOutputs[l][i])
		}
	}

	for l := numLayers - 2; l >= 0; l-- {
		for i := range weights[l] {
			for j := range weights[l][i] {
				weights[l][i][j] += learningRate * delta[l+1][i] * layerOutputs[l][j]
			}
			biases[l][i] += learningRate * delta[l+1][i]
		}
	}
}

// Feedforward
func feedForward() float64 {
	layerOutputs[0] = inputs

	for l := 0; l < len(weights); l++ {
		for i := range weights[l] {
			input := 0.0
			for j := range weights[l][i] {
				input += layerOutputs[l][j] * weights[l][i][j]
			}
			input += biases[l][i]

			activated := relu(input)

			if l != len(weights)-1 { // Only apply dropout on hidden layers
				if rand.Float64() < dropoutProb {
					layerOutputs[l+1][i] = 0
				} else {
					layerOutputs[l+1][i] = activated
				}
			} else {
				// Output layer - no dropout
				layerOutputs[l+1][i] = activated
			}
		}
	}

	predicted = layerOutputs[len(layerOutputs)-1][0]

	return predicted
}

func main() {
	layerSizes := []int{2, 3, 2, 1}
	makeNN(layerSizes)
	dropoutProb = 0 // For anything less precise than addition, set dropout % accordingly

	for {
		target = inputs[0] + inputs[1]
		feedForward()
		backprop()
		if math.Abs(cost()) <= 0.0001 {
			inputs[0] = rand.Float64()*9 + 1
			inputs[1] = rand.Float64()*9 + 1
			target = inputs[0] + inputs[1]
			feedForward()
			if math.Abs(cost()) <= 0.00001 {
				fmt.Println("Finished without learning")
			} else {
				fmt.Println("Not correct")
			}
		}
	}
}
