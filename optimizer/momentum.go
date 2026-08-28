package optimizer

import (
	"github.com/itsubaki/autograd/tensor"
)

// Momentum is an optimizer that uses momentum-based gradient descent.
type Momentum struct {
	LearningRate float64
	Momentum     float64
	Hook         []Hook
	Vs           map[string]*tensor.Tensor[float64]
}

// Update updates the parameters of the model.
func (o *Momentum) Update(model Model) {
	if len(o.Vs) == 0 {
		o.Vs = make(map[string]*tensor.Tensor[float64])
	}

	params := Params(model, o.Hook)
	for name, p := range params {
		if _, ok := o.Vs[name]; !ok {
			o.Vs[name] = tensor.ZeroLike(p.Data)
		}

		// param = param + (momentum * v - lr * grad)
		o.Vs[name] = tensor.F2(o.Vs[name], p.Grad.Data, momentum(o.Momentum, o.LearningRate))
		p.Data = tensor.Add(p.Data, o.Vs[name])
	}
}

// momentum returns a function that computes the momentum update for a given velocity v and gradient grad.
func momentum(momentum, lr float64) func(v, grad float64) float64 {
	return func(v, grad float64) float64 { return momentum*v - lr*grad }
}
