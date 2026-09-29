package optimizer

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/tensor"
)

// Momentum is an optimizer that uses momentum-based gradient descent.
type Momentum struct {
	LearningRate float32
	Momentum     float32
	Vs           map[string]*tensor.Tensor[float32]
}

// Update updates the parameters of the model.
func (o *Momentum) Update(params layer.Parameters) {
	if len(o.Vs) == 0 {
		o.Vs = make(map[string]*tensor.Tensor[float32])
	}

	for name, p := range params {
		if p.Grad == nil {
			continue
		}

		if _, ok := o.Vs[name]; !ok {
			o.Vs[name] = tensor.ZerosLike(p.Data)
		}

		// param = param + (momentum * v - lr * grad)
		o.Vs[name] = tensor.F2(o.Vs[name], p.Grad.Data, momentum(o.Momentum, o.LearningRate))
		p.Data = tensor.Add(p.Data, o.Vs[name])
	}
}

// momentum returns a function that computes the momentum update for a given velocity v and gradient grad.
func momentum(momentum, lr float32) func(v, grad float32) float32 {
	return func(v, grad float32) float32 { return momentum*v - lr*grad }
}
