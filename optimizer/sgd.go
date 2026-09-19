package optimizer

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/tensor"
)

// SGD is an optimizer that uses the stochastic gradient descent algorithm.
type SGD struct {
	LearningRate float32
}

// Update updates the parameters of the model.
func (o *SGD) Update(params layer.Parameters) {
	for _, p := range params {
		if p.Grad == nil {
			continue
		}

		p.Data = tensor.F2(p.Data, p.Grad.Data, sgd(o.LearningRate))
	}
}

// sgd returns a function that computes the SGD update for a given parameter value a and gradient b using the specified learning rate lr.
func sgd(lr float32) func(a, b float32) float32 {
	return func(a, b float32) float32 { return a - lr*b }
}
