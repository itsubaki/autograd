package optimizer

import "github.com/itsubaki/autograd/tensor"

// SGD is an optimizer that the Stochastic Gradient Descent algorithm.
type SGD struct {
	LearningRate float64
	Hook         []Hook
}

// Update updates the parameters of the model.
func (o *SGD) Update(model Model) {
	params := Params(model, o.Hook)

	for _, p := range params {
		p.Data = tensor.F2(p.Data, p.Grad.Data, sgd(o.LearningRate))
	}
}

func sgd(lr float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return a - lr*b }
}
