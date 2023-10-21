package optimizer

import "github.com/itsubaki/autograd/matrix"

// SGD is an optimizer that the Stochastic Gradient Descent algorithm.
type SGD struct {
	LearningRate float64
	Hooks        []Hook
}

// Update updates the parameters of the model.
func (o *SGD) Update(model Model) {
	params := Params(model)
	for _, h := range o.Hooks {
		h(params)
	}

	for _, p := range params {
		p.Data = matrix.F2(p.Data, p.Grad.Data, sgd(o.LearningRate))
	}
}

func sgd(learningRate float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return a - learningRate*b }
}
